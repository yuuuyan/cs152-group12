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
        self.guild_id = None

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
            self.reports[author_id] = Report(self)

        # Let the report class handle this message; forward all the messages it returns to uss
        responses = await self.reports[author_id].handle_message(message)
        for r in responses:
            await message.channel.send(r)
        
        # send message to moderator channel begining review process
        if self.reports[author_id].report_awaiting_review():
            mod_channel = self.mod_channels[self.guild_id] # hard coded; TODO: change this somehow...
            mod_message = "Report %s is ready for review..." % (author_id)
            await mod_channel.send([mod_message])


        # If the report is complete or cancelled, remove it from our map
        if self.reports[author_id].report_complete():
            self.reports.pop(author_id)

    async def handle_channel_message(self, message):
        # Only handle messages sent in the "group-#-mod" channel
        if not message.channel.name == f'group-{self.group_num}-mod':
            return

        # Forward the message to the mod channel
        mod_channel = self.mod_channels[self.guild.id]
        # await mod_channel.send(f'Forwarded message:\n{message.author.name}: "{message.content}"')
        # scores = self.eval_text(message.content)
        # await mod_channel.send(self.code_format(scores))

        if "MOD_REVIEW" in message.content:
            init_review_text = message.content.strip().split()
            author_id = init_review_text[1]
            if author_id not in self.reports.keys() or self.reports[author_id].report_complete():
                reply = "Invalid report ID %s mentioned for moderator review. " % (author_id)
                reply += "Please restart review process with the correct report ID."
                await mod_channel.send(reply)
            else:
                self.reports[author_id].print_moderator_summary()
                # TODO: fill in available actions with numbers
                reply = "Available actions include: "
                await mod_channel.send(reply)

        elif "SHOW_REPORTS" in message.content:
            reply = "Available reports are: "
            for author_id in self.reports.keys():
                if not self.reports[author_id].report_complete():
                    reply += author_id + " "
            await mod_channel.send(reply)


        elif "TAKE_ACTION" in message.content:
            init_action_text = message.content.strip().split()
            if len(init_action_text) != 3:
                reply = "Invalid format for TAKE_ACTION. Expected \"TAKE_ACTION REPORT_ID ACTION_NUMBER\""
            elif init_action_text[2] not in self.reports.keys() or self.reports[init_action_text[2]].report_complete():
                reply = "Invalid report ID %s provided for TAKE_ACTION" % (init_action_text[2])
            elif init_action_text[3] not in []:
                # TODO: replace empty list with list of integer strings representing valid actions
                reply = "Invalid action %s provided for TAKE_ACTION" % (init_action_text[3])
            else:
                # TODO: implement moderator action based on report
                # note: cannot actually delete users so just send them a direct message saying they are deleted isntead
                # report ID is the same as author_ID which can help in sending them a direct message

                # mark report as completed after executing action and pop from report map
                self.reports[init_action_text[2]].mark_completed()
                self.reports.pop(author_id)





    
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