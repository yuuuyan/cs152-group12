# bot.py
import discord
from discord.ext import commands
import os
import json
import logging
import re
import requests
from report import Report, AutomaticReport
import pdb
from collections import defaultdict
from dataclasses import dataclass
from typing import Optional
import vertexai
from vertexai.generative_models import GenerativeModel
from together import Together
from enum import Enum

# Set up logging to the console
logger = logging.getLogger('discord')
logger.setLevel(logging.DEBUG)
handler = logging.FileHandler(filename='discord.log', encoding='utf-8', mode='w')
handler.setFormatter(logging.Formatter('%(asctime)s:%(levelname)s:%(name)s: %(message)s'))
logger.addHandler(handler)

NUM_FLAG_AUTOMATIC = 3

# GEMINI, LLAMA_70B, LLAMA_8B, CLASSIFIER
class LLM_Moderator(Enum):
    gemini = 1
    llama70b = 2
    llama8b = 3
    classifier = 4
    everything = 5
    nothing = 6

    @staticmethod
    def from_str(label):
        if label == "gemini":
            return LLM_Moderator.gemini
        elif label == "llama70b":
            return LLM_Moderator.llama70b
        elif label == "llama8b":
            return LLM_Moderator.llama8b
        elif label == "classifier":
            return LLM_Moderator.classifier
        elif label == "everything":
            return LLM_Moderator.everything
        elif label == "nothing":
            return LLM_Moderator.nothing
        else:
            raise NotImplementedError("Invalid llm moderation policy %s provided" % (label))


# There should be a file called 'tokens.json' inside the same folder as this file
token_path = 'tokens.json'
if not os.path.isfile(token_path):
    raise Exception(f"{token_path} not found!")
with open(token_path) as f:
    # If you get an error here, it means your token is formatted incorrectly. Did you put it in quotes?
    tokens = json.load(f)
    discord_token = tokens['discord']
    TOGETHER_AI_API_KEY = tokens['together-ai']
    GCP_PROJECT_ID = tokens['gcp']
    ml_moderator = LLM_Moderator.from_str(tokens['llm-moderator'].lower())
prompt_path =  'LLM_prompt.txt'
with open(prompt_path) as f:
    LLM_PROMPT = f.read()

@dataclass
class UserStats:
    name: Optional[str] = None
    total_reports: int = 0
    num_malicious_reports: int = 0
    num_suspensions: int = 0
    num_posts_deleted: int = 0
    automatic_post_deletions: int = 0


class ModBot(discord.Client):
    def __init__(self): 
        intents = discord.Intents.default()
        intents.message_content = True
        super().__init__(command_prefix='.', intents=intents)
        self.group_num = None
        self.mod_channels = {} # Map from guild to the mod channel id for that guild
        self.reports = {} # Map from user IDs to the state of their report
        self.submitted_reports = {} # Map from (userID + msgID) to report
        self.guild_id = None
        self.username_map = {} # map of username to user id
        self.auto_deleted_msgs = defaultdict(list) # dictionary from abuser_id to list of auto deleted messages

        # map from user names to statistics about user regarding moderation
        self.user_stats = defaultdict(UserStats)

        self.ActionDict = {1: "Determine malicous report", 2: "Suspend account indefinitely", 
                           3: "Suspend account for some time", 4: "Delete content and warn user", 
                           5: "Close report without any action"}
        self.mod_actions_auto = [2, 3, 5] # valid actions for automatic report

    async def on_ready(self):
        print(f'{self.user.name} has connected to Discord! It is these guilds:')
        for guild in self.guilds:
            print(f' - {guild.name}')
        print('Press Ctrl-C to quit.')

        # Parse the group number out of the bot's name
        match = re.search('[gG]roup (\d+) [bB]ot', self.user.name)
        if match:
            self.group_num = match.group(1)
        else:
            raise Exception("Group number not found in bot's name. Name format should be \"Group # Bot\".")

        # Find the mod channel in each guild that this bot should report to
        for guild in self.guilds:
            for channel in guild.text_channels:
                if channel.name == f'group-{self.group_num}-mod':
                    self.mod_channels[guild.id] = channel
                    self.guild_id = guild.id
        

    async def on_message(self, message):
        '''
        This function is called whenever a message is sent in a channel that the bot can see (including DMs). 
        Currently the bot is configured to only handle messages that are sent over DMs or in your group's "group-#" channel. 
        '''
        # Ignore messages from the bot 
        if message.author.id == self.user.id:
            return

        # Check if this message was sent in a server ("guild") or if it's a DM
        if message.guild:
            await self.handle_channel_message(message)
        else:
            await self.handle_dm(message)

    async def handle_dm(self, message):
        # Handle a help message
        if message.content == Report.HELP_KEYWORD:
            reply =  "Use the `report` command to begin the reporting process.\n"
            reply += "Use the `cancel` command to cancel the report process.\n"
            await message.channel.send(reply)
            return

        author_id = message.author.id
        responses = []

        # Only respond to messages if they're part of a reporting flow
        if author_id not in self.reports and not message.content.startswith(Report.START_KEYWORD):
            return

        # If we don't currently have an active report for this user, add one
        if author_id not in self.reports:
            self.reports[author_id] = Report(self, author_id=author_id, submitted_reports=self.submitted_reports)

        # Let the report class handle this message; forward all the messages it returns to us
        responses = await self.reports[author_id].handle_message(message)
        for r in responses:
            await message.channel.send(r)
        
        # send message to moderator channel begining review process
        if self.reports[author_id].report_awaiting_review():

            if author_id not in self.user_stats.keys() or len(self.user_stats[author_id].name) == 0:
                self.user_stats[author_id].name = author_id

            self.username_map[self.reports[author_id].author_name] = self.reports[author_id].abuser_id
            self.username_map[message.author.name] = author_id

            # incrementing report count
            self.user_stats[author_id].total_reports += 1

            mod_channel = self.mod_channels[self.guild_id] # hard coded; TODO: change this somehow...
            report_seq = str(author_id) + str(self.reports[author_id].reported_message_id)
            mod_message = "Report %s is ready for review..." % (report_seq)
            report_to_review =  self.reports[author_id]
            self.submitted_reports[report_seq] = report_to_review
            self.reports.pop(author_id)
            await mod_channel.send(mod_message)
        # If the report is complete or cancelled, remove it from our map
        elif self.reports[author_id].report_complete():
            self.reports.pop(author_id)

    async def handle_channel_message(self, message):
        mod_channel = self.mod_channels[self.guild_id]
        # Handle messages sent outside of mod channel
        if not message.channel.name == f'group-{self.group_num}-mod':
            # Evaluate text
            found_malicious_message = self.eval_text(message.content)

            if found_malicious_message:
                abuser_id = message.author.id
                abuser_name = message.author.name
                msg_content = message.content

                try:
                    await message.delete()
                    abuser = await self.fetch_user(abuser_id)
                    abuser_msg = f"WARNING: Your account may be suspended if you continue to post content in violation of our community guidelines. "
                    abuser_msg += "We deleted the following message from our platform since it violated our community guidelines: %s" % (msg_content)
                    await abuser.send(abuser_msg)
                except:
                    return

                # updating user stats
                self.username_map[abuser_name] = abuser_id
                self.user_stats[abuser_id].num_posts_deleted += 1
                self.user_stats[abuser_id].automatic_post_deletions += 1

                # storing history of deleted messages
                self.auto_deleted_msgs[abuser_id].append(msg_content)

                if self.user_stats[abuser_id].automatic_post_deletions % NUM_FLAG_AUTOMATIC == 0:
                    # retrieve deleted messages
                    deleted_msgs = self.auto_deleted_msgs[abuser_id]

                    # create report for moderator for automatic flagging
                    report = AutomaticReport(client, self.submitted_reports)
                    report.create_report(abuser_id, abuser_name, self.user_stats[abuser_id].automatic_post_deletions, deleted_msgs)

                    # inform moderator that it is available for review
                    mod_message = "Automatic report %s is ready for review..." % (abuser_id)
                    await mod_channel.send(mod_message)
            return
        else:
            # in moderator channel, everything in channels outside moderator channels
            # Handle messages in mod channel
            # Forward the message to the mod channel
            
            # await mod_channel.send(f'Forwarded message:\n{message.author.name}: "{message.content}"')
            # scores = self.eval_text(message.content)
            # await mod_channel.send(self.code_format(scores))
            # await mod_channel.send(self.eval_text(message.content))

            if "help" == message.content.strip().lower():
                reply = "Available actions (case-sensitive) include: \n"
                reply += "SHOW_REPORTS: Show all reports available for review\n"
                reply += "MOD_REVIEW <REPORT ID>: Display moderator summmary for mentioned report\n"
                reply += "TAKE_ACTION <REPORT ID> <ACTION ID>: Take the mentioned action on the given report and close report\n"
                reply += "USER_STATS <USERNAME>: Print out statistics about user's moderation history\n"
                reply += "PRINT_USERS: List usernames which have a moderation record associated with them\n"

                await mod_channel.send(reply)
            elif "MOD_REVIEW" in message.content:
                init_review_text = message.content.strip().split()
                if len(init_review_text) != 2:
                    reply = "Incorrect format for command MOD_REVIEW. Expected \"MOD_REVIEW <REPORT ID>\"."
                    await mod_channel.send(reply)
                    return
                author_id = int(init_review_text[1])
                dtype_key = type(list(self.submitted_reports.keys())[0]) if len(self.submitted_reports) > 0 else None
                if dtype_key is not None:
                    try:
                        author_id = dtype_key(author_id)
                    except:
                        pass
                if author_id not in self.submitted_reports.keys() or self.submitted_reports[author_id].report_complete():
                    if author_id in self.submitted_reports and self.submitted_reports[self.submitted_reports.pop(author_id)].report_complete():
                        self.submitted_reports.pop(author_id)
                    reply = "Invalid report ID %s mentioned for moderator review. " % (author_id)
                    reply += "Please restart review process with the correct report ID."
                    await mod_channel.send(reply)
                else:
                    if self.submitted_reports[author_id].report_type != "automatic":
                        reporter = await self.fetch_user(self.submitted_reports[author_id].author_id)
                    else:
                        reporter = None
                    reporter_name = reporter.name if reporter else "DISCORD_BOT"
                    reply = f"Reporter: {reporter_name}; " + self.submitted_reports[author_id].print_moderator_summary()[0] + "\n"
                    reply += "Available actions include (ID : ACTION): \n"

                    if self.submitted_reports[author_id].report_type != "automatic":
                        reply += "".join([str(key) + " : " + value + "\n" for key,value in self.ActionDict.items()])
                    else:
                        reply += "".join([str(key) + " : " + value + "\n" for key,value in self.ActionDict.items() if key in self.mod_actions_auto])
                    await mod_channel.send(reply)

            elif "SHOW_REPORTS" in message.content:
                reply = "Available reports are: "

                for author_id in self.submitted_reports.keys():
                    if not self.submitted_reports[author_id].report_complete():
                        reply += str(author_id) + " "
                    else:
                        self.submitted_reports.pop(author_id)
                await mod_channel.send(reply)


            elif "TAKE_ACTION" in message.content:

                init_action_text = message.content.strip().split()
                actions_str = list(self.ActionDict.keys())
                actions_str = [str(x) for x in actions_str]
                valid_actions_auto = [str(x) for x in self.mod_actions_auto]
                if len(init_action_text) != 3:
                    reply = "Invalid format for TAKE_ACTION. Expected \"TAKE_ACTION REPORT_ID ACTION_ID\"."
                elif init_action_text[1] not in self.submitted_reports.keys() or self.submitted_reports[init_action_text[1]].report_complete():
                    if init_action_text[1] in self.submitted_reports and self.submitted_reports[init_action_text[1]].report_complete():
                        self.submitted_reports.pop(init_action_text[1])
                    reply = "Invalid report ID %s provided for TAKE_ACTION" % (init_action_text[1])
                elif init_action_text[2] not in actions_str:
                    reply = "Invalid action %s provided for TAKE_ACTION" % (init_action_text[2])
                elif self.submitted_reports[init_action_text[1]].report_type == "automatic" and init_action_text[2] not in valid_actions_auto:
                    reply = "Invalid action %s provided for automatic report in TAKE_ACTION" % (init_action_text[2])
                else:
                    code = init_action_text[2]
                    mal_reporter = False
                    report_id = init_action_text[1]
                    abuser = await self.fetch_user(self.submitted_reports[report_id].abuser_id)
                    auto_flag = False

                    if self.submitted_reports[report_id].author_id is not None:
                        reporter = await self.fetch_user(self.submitted_reports[report_id].author_id)
                    elif self.submitted_reports[report_id].report_type == "automatic":
                        reporter = None
                        auto_flag = True

                    if (reporter and abuser) or (abuser and auto_flag):
                        abuser_id = self.submitted_reports[report_id].abuser_id

                        if not auto_flag:
                            reporter_id = self.submitted_reports[report_id].author_id

                        # filling metadata for printing user statistics - only need to track here
                        #  as otherwise can print default stats since no action was taken to change defaults
                        self.username_map[abuser.name] = abuser_id
                        if not auto_flag:
                            self.username_map[reporter.name] = reporter_id

                        # did not store this when starting stats for reporter
                        self.user_stats[abuser.id].name = abuser.name

                        if code == "1":
                            mal_reporter = True
                            response_content = None

                            if self.user_stats[reporter_id].num_malicious_reports > 3:
                                response_content = 'n' # not first three offenses
                            else:
                                response_content = 'y' # is a first three offense

                            self.user_stats[reporter_id].num_malicious_reports += 1

                            # automating this based on tracked statistics
                            # reply = "Is this a first-time offense for the reporter? Please reply \"y\" or \"n\""
                            # await mod_channel.send(reply)
                            
                            # # Wait for moderator response
                            # def check(response_message):
                            #     return response_message.author == message.author and response_message.channel == mod_channel
                            
                            # response_message = await self.wait_for('message', check=check)
                            # response_content = response_message.content.lower().strip()
                            
                            if response_content == 'y':
                                code = "MALICIOUS_WARNING"
                            elif response_content == 'n':
                                code = "3"
                            else:
                                reply = "Invalid entry"
                        if code == "MALICIOUS_WARNING": # no need in action_dict since we automatically track this in statistics now
                            # Send warning DM to reporter
                            try:
                                await reporter.send("WARNING: Your account may be suspended if you continue to create malicious reports.")
                                reply = f"Warning sent to {reporter.name}"
                            except discord.Forbidden:
                                reply = "I do not have permissions to send a DM."
                        if code == "2": # indefinite suspension
                            try:
                                await abuser.send("ATTENTION: Your account has been indefinitely suspended for violating our community guidelines.")
                                reply = f"The account of {abuser.name} has been suspended indefinitely"
                                self.user_stats[abuser_id].num_suspensions += 1
                            except discord.Forbidden:
                                reply = "I do not have permissions to send a DM."
                        if code == "3": # suspension of mentioned duration
                            reply = ""
                            if mal_reporter:
                                reply += "This account has created %d malicious reports previously. " % (self.user_stats[reporter_id].num_malicious_reports - 1)
                            reply += "How long should their account be suspended for? (e.g. 60 hrs)"
                            await mod_channel.send(reply)

                            # Wait for moderator response
                            def check(response_message):
                                return response_message.author == message.author and response_message.channel == mod_channel
                            
                            response_message = await self.wait_for('message', check=check)
                            time = response_message.content.lower().strip()
                            
                            if mal_reporter:
                                try:
                                    await reporter.send(f"ATTENTION: Your account has been suspended for malicous reporting for {time}.")
                                    self.user_stats[reporter_id].num_suspensions += 1
                                    reply = f"The account of {reporter.name} has been suspended for {time}."
                                except discord.Forbidden:
                                    reply = "I do not have permissions to send a DM."
                            else:
                                try:
                                    await abuser.send(f"ATTENTION: Your account has been suspended for {time} for violating our community guidelines.")
                                    self.user_stats[abuser_id].num_suspensions += 1
                                    reply = f"The account of {abuser.name} has been suspended for {time}."
                                except discord.Forbidden:
                                    reply = "I do not have permissions to send a DM."

                        # in addition to suspending account, delete reported message when suspension is handed out
                        if (code == "2" or code == "3") and self.submitted_reports[report_id].report_type != "automatic":
                            try:
                                await self.submitted_reports[report_id].message.delete()
                            except:
                                pass

                        if code == "4": # delete content
                            msg_content = self.submitted_reports[report_id].message.content
                            try:
                                await self.submitted_reports[report_id].message.delete()
                                reply = "The following message has been deleted from the server: %s" % (msg_content)
                                self.user_stats[abuser_id].num_posts_deleted += 1
                                try:
                                    abuser_msg = f"WARNING: Your account may be suspended if you continue to post content in violation of our community guidelines. "
                                    abuser_msg += "We deleted the following message from our platform since it violated our community guidelines: %s" % (msg_content)
                                    await abuser.send(abuser_msg)
                                    reply += f" and a warning has been sent to {abuser.name}."
                                except discord.Forbidden:
                                    reply += " Could not send a warning, I do not haver permissions to send a DM."
                            except:
                                reply = "The following message no longer exists on our server at the time of review: %s" % (msg_content)

                        if code == "5": # do nothing
                            reply = "Report %s has been closed without taking action." % (report_id)

                    # mark report as completed after executing action and pop from report map
                    self.submitted_reports[report_id].mark_completed()
                    self.submitted_reports.pop(report_id)

                await mod_channel.send(reply)

            elif 'USER_STATS' in message.content:
                msg = message.content.strip().split()
                if len(msg) != 2:
                    reply = "Invalid format for USER_STATS. Expected \"USER_STATS USERNAME\""
                else:
                    username = msg[1]

                    if username in self.username_map.keys():
                        user_id = self.username_map[username]
                        is_found = await self.fetch_user(user_id)
                    else:
                        is_found = None

                    if is_found is None:
                        user_stats = UserStats()
                    else:
                        user_stats = self.user_stats[user_id]
                    reply = "Here are the statistics for username %s:\n" % (username)
                    reply += "Total reports: %d\n" % (user_stats.total_reports) 
                    reply += "Malicious reports: %d\n" % (user_stats.num_malicious_reports) 
                    reply += "Number of suspensions: %d\n" % (user_stats.num_suspensions)
                    reply += "Number of posts deleted: %d\n" %  (user_stats.num_posts_deleted)
                    reply += "Number of posts deleted by bot: %d\n" % (user_stats.automatic_post_deletions)

                await mod_channel.send(reply)
            
            elif 'PRINT_USERS' in message.content:
                if len(message.content.strip().split()) != 1:
                    reply = "Invalid format for PRINT_USERS. Expected \"PRINT_USERS\""
                else:
                    available_users = list(self.username_map)
                    reply = "We have records for the following %d users:\n" % (len(available_users))
                    for e, u in enumerate(available_users):
                        reply += u
                        if e != len(available_users) - 1:
                            reply += ", "
                await mod_channel.send(reply)

    
    def eval_text(self, message):
        ''''
        TODO: Once you know how you want to evaluate messages in your channel, 
        insert your code here! This will primarily be used in Milestone 3. 
        '''

        try:
            if ml_moderator == LLM_Moderator.llama70b:
                response = self.query_together_ai("meta-llama/Llama-3-70b-chat-hf", LLM_PROMPT + message)
            elif ml_moderator == LLM_Moderator.llama8b:
                response = self.query_together_ai("meta-llama/Llama-3-8b-chat-hf", LLM_PROMPT + message)
            elif ml_moderator == LLM_Moderator.gemini:
                response = self.query_gcp(LLM_PROMPT + message)
            elif ml_moderator == LLM_Moderator.classifier:
                raise NotImplementedError("Classifier not implemented for moderation yet")
                response = None
            elif ml_moderator == LLM_Moderator.nothing:
                response = "Misinformation: no"
            elif ml_moderator == LLM_Moderator.everything:
                response = "Misinformation: yes"
            else:
                raise NotImplementedError("Moderator policy not implemented")
            
            
            response = response.lower().strip()
            try:
                response = response[16:]
            except:
                response = "no"
            
            if response == "yes":
                response = True # contains misinformation/disinformation, etc
            else:
                response = False # not to be moderated if invalid response or "no"
        except:
            response = False # if invalid response or error encountered, do nothing

        
        return response
        

    
    def code_format(self, text):
        ''''
        TODO: Once you know how you want to show that a message has been 
        evaluated, insert your code here for formatting the string to be 
        shown in the mod channel. 
        '''
        return "Evaluated: '" + text+ "'"

    def query_together_ai(self, model: str, text: str) -> dict:
        client_together = Together(api_key=TOGETHER_AI_API_KEY)

        response = client_together.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": text}]
        )

        return response.choices[0].message.content


    def query_gcp(self, text: str) -> dict:
        vertexai.init(project=GCP_PROJECT_ID, location="us-central1")
        model = GenerativeModel(model_name="gemini-1.0-pro-002")

        response = model.generate_content(
            text
        )

        return response.text

client = ModBot()
client.run(discord_token)