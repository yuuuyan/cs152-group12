# bot.py
import discord
from discord.ext import commands
import os
import json
import logging
import re
import requests
from report import Report
import pdb
from collections import defaultdict

# Set up logging to the console
logger = logging.getLogger('discord')
logger.setLevel(logging.DEBUG)
handler = logging.FileHandler(filename='discord.log', encoding='utf-8', mode='w')
handler.setFormatter(logging.Formatter('%(asctime)s:%(levelname)s:%(name)s: %(message)s'))
logger.addHandler(handler)

# There should be a file called 'tokens.json' inside the same folder as this file
token_path = 'tokens.json'
if not os.path.isfile(token_path):
    raise Exception(f"{token_path} not found!")
with open(token_path) as f:
    # If you get an error here, it means your token is formatted incorrectly. Did you put it in quotes?
    tokens = json.load(f)
    discord_token = tokens['discord']


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

        self.ActionDict = {1: "Determine malicous report", 2: "Warn reporter", 3: "Suspend account indefinitely", 
                           4: "Suspend account for some time", 5: "Delete content and warn user"}

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

        # Let the report class handle this message; forward all the messages it returns to uss
        responses = await self.reports[author_id].handle_message(message)
        for r in responses:
            await message.channel.send(r)
        
        # send message to moderator channel begining review process
        if self.reports[author_id].report_awaiting_review():
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
        # Only handle messages sent in the "group-#-mod" channel
        if not message.channel.name == f'group-{self.group_num}-mod':
            return

        # Forward the message to the mod channel
        mod_channel = self.mod_channels[self.guild_id]
        # await mod_channel.send(f'Forwarded message:\n{message.author.name}: "{message.content}"')
        # scores = self.eval_text(message.content)
        # await mod_channel.send(self.code_format(scores))

        if "help" == message.content.strip().lower():
            reply = "Available actions (case-sensitive) include: \n"
            reply += "SHOW_REPORTS: Show all reports available for review\n"
            reply += "MOD_REVIEW <REPORT ID>: Display moderator summmary for mentioned report\n"
            reply += "TAKE_ACTION <REPORT ID> <ACTION ID>: Take the mentioned action on the given report and close report" 
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
                reporter = await self.fetch_user(self.submitted_reports[author_id].author_id)
                reply = f"Reporter: {reporter.name}; " + self.submitted_reports[author_id].print_moderator_summary()[0] + "\n"
                reply += "Available actions include (ID : ACTION): \n"
                reply += "".join([str(key) + " : " + value + "\n" for key,value in self.ActionDict.items()])
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
            if len(init_action_text) != 3:
                reply = "Invalid format for TAKE_ACTION. Expected \"TAKE_ACTION REPORT_ID ACTION_ID\"."
            elif init_action_text[1] not in self.submitted_reports.keys() or self.submitted_reports[init_action_text[1]].report_complete():
                if init_action_text[1] in self.submitted_reports and self.submitted_reports[init_action_text[1]].report_complete():
                    self.submitted_reports.pop(init_action_text[1])
                reply = "Invalid report ID %s provided for TAKE_ACTION" % (init_action_text[1])
            elif init_action_text[2] not in actions_str:
                reply = "Invalid action %s provided for TAKE_ACTION" % (init_action_text[2])
            else:
                code = init_action_text[2]
                mal_reporter = False
                report_id = init_action_text[1]
                abuser = await self.fetch_user(self.submitted_reports[report_id].abuser_id)
                reporter = await self.fetch_user(self.submitted_reports[report_id].author_id)
                if reporter and abuser:
                    if code == "1":
                        mal_reporter = True
                        reply = "Is this a first-time offense for the reporter? Please reply \"y\" or \"n\""
                        await mod_channel.send(reply)
                        
                        # Wait for moderator response
                        def check(response_message):
                            return response_message.author == message.author and response_message.channel == mod_channel
                        
                        response_message = await self.wait_for('message', check=check)
                        response_content = response_message.content.lower().strip()
                        
                        if response_content == 'y':
                            code = "2"
                        elif response_content == 'n':
                            code = "4"
                        else:
                            reply = "Invalid entry"
                    if code == "2":
                        # Send warning DM to reporter
                        try:
                            await reporter.send("WARNING: Your account may be suspended if you continue to create malicious reports.")
                            reply = f"Warning sent to {reporter.name}"
                        except discord.Forbidden:
                            reply = "I do not have permissions to send a DM."
                    if code == "3":
                        try:
                            await abuser.send("ATTENTION: Your account has been indefinitely suspended for violating our community guidelines.")
                            reply = f"The account of {abuser.name} has been suspended indefinitely"
                        except discord.Forbidden:
                            reply = "I do not have persmissions to send a DM."
                    if code == "4":
                        reply = "How long should their account be suspended for? (e.g. 60hrs)"
                        await mod_channel.send(reply)

                        # Wait for moderator response
                        def check(response_message):
                            return response_message.author == message.author and response_message.channel == mod_channel
                        
                        response_message = await self.wait_for('message', check=check)
                        time = response_message.content.lower().strip()
                        
                        if mal_reporter:
                            try:
                                await reporter.send(f"ATTENTION: Your account has been suspended for malicous reporting for {time}.")
                                reply = f"The account of {reporter.name} has been suspended for {time}."
                            except discord.Forbidden:
                                reply = "I do not have permissions to send a DM."
                        else:
                            try:
                                await abuser.send(f"ATTENTION: Your account has been suspended for {time} for violating our community guidelines.")
                                reply = f"The account of {abuser.name} has been suspended for {time}."
                            except discord.Forbidden:
                                reply = "I do not have permissions to send a DM."

                    if code == "5":
                        msg_content = self.submitted_reports[report_id].message.content
                        try:
                            await self.submitted_reports[report_id].message.delete()
                            reply = "The following message has been deleted from the server: %s" % (msg_content)
                            try:
                                await abuser.send(f"WARNING: Your account may be suspended if you continue to post content in violation of our community guidelines.")
                                reply += f" and a warning has been sent to {abuser.name}."
                            except discord.Forbidden:
                                reply += " Could not send a warning, I do not haver permissions to send a DM."
                        except:
                            reply = "The following message no longer exists on our server at the time of review: %s" % (msg_content)

                # mark report as completed after executing action and pop from report map
                self.submitted_reports[report_id].mark_completed()
                self.submitted_reports.pop(report_id)

            await mod_channel.send(reply)
    
    def eval_text(self, message):
        ''''
        TODO: Once you know how you want to evaluate messages in your channel, 
        insert your code here! This will primarily be used in Milestone 3. 
        '''
        return message

    
    def code_format(self, text):
        ''''
        TODO: Once you know how you want to show that a message has been 
        evaluated, insert your code here for formatting the string to be 
        shown in the mod channel. 
        '''
        return "Evaluated: '" + text+ "'"


client = ModBot()
client.run(discord_token)