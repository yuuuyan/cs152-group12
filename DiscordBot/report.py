from enum import Enum, auto
import discord
import re

class State(Enum):
    REPORT_START = auto()
    AWAITING_MESSAGE = auto()
    MESSAGE_IDENTIFIED = auto()
    REPORT_COMPLETE = auto()
    
    # general options added
    REPORT_CANCELLED = auto()
    ADDITIONAL_INFO = auto()
    AWAITING_REVIEW = auto()

    # impersonation flow
    IMPERSONATION_INIT = auto()
    YES_IMPERSONATION = auto()
    WHO_IMPERSONATION = auto()
    USER_IMPERSONATED = auto()
    SOMEONE_IMPERSONATED = auto()


    # misinformation flow
    MISINFORMATION_INIT = auto()
    YES_MISINFORMATION = auto()
    MISINFORMATION_TYPE = auto()

class Report:
    START_KEYWORD = "report"
    CANCEL_KEYWORD = "cancel"
    HELP_KEYWORD = "help"

    def __init__(self, client, author_id=None, submitted_reports=None):
        self.state = State.REPORT_START
        self.client = client
        self.message = None

        # for verification in impersonation flow
        self.author_name = None
        self.abuser_id = None

        # for terminating reporting process if wrong options are chosen
        self.num_attempts = 0

        # keeping track of submitted reports
        self.submitted_reports = submitted_reports
        self.author_id = author_id
        self.reported_message_id = None

        # summary string of report
        self.summary = ""
    
    async def handle_message(self, message):
        '''
        This function makes up the meat of the user-side reporting flow. It defines how we transition between states and what 
        prompts to offer at each of those states. You're welcome to change anything you want; this skeleton is just here to
        get you started and give you a model for working with Discord. 
        '''

        if message.content == self.CANCEL_KEYWORD:
            self.state = State.REPORT_COMPLETE
            return ["Report cancelled."]
        
        if self.state == State.REPORT_START:
            reply =  "Thank you for starting the reporting process. "
            reply += "Say `help` at any time for more information.\n\n"
            reply += "Please copy paste the link to the message you want to report.\n"
            reply += "You can obtain this link by right-clicking the message and clicking `Copy Message Link`."
            reply += "If you want to report a user for impersonation, please feel free to pick an arbitrary message sent by them."
            self.state = State.AWAITING_MESSAGE
            return [reply]
        
        if self.state == State.AWAITING_MESSAGE:
            # Parse out the three ID strings from the message link
            m = re.search('/(\d+)/(\d+)/(\d+)', message.content)
            if not m:
                return ["I'm sorry, I couldn't read that link. Please try again or say `cancel` to cancel."]
            guild = self.client.get_guild(int(m.group(1)))
            if not guild:
                return ["I cannot accept reports of messages from guilds that I'm not in. Please have the guild owner add me to the guild and try again."]
            channel = guild.get_channel(int(m.group(2)))
            if not channel:
                return ["It seems this channel was deleted or never existed. Please try again or say `cancel` to cancel."]
            try:
                message = await channel.fetch_message(int(m.group(3)))
            except discord.errors.NotFound:
                return ["It seems this message was deleted or never existed. Please try again or say `cancel` to cancel."]
            
            self.reported_message_id = message.id
            if self.reported_message_id is not None and str(self.author_id) + str(self.reported_message_id) in self.submitted_reports:
                self.state = State.REPORT_CANCELLED
                return ["There already exists a report from you for this message. Please wait for that to be resolved before starting another report."]

            # Here we've found the message - it's up to you to decide what to do next!
            self.state = State.MESSAGE_IDENTIFIED
            reply = "Would you like to report the user for impersonation or for spreading misinformation? "
            reply += "Type \"impersonation\" for reporting impersonation and \"misinformation\" for reporting misinformation"

            self.message = message.content
            self.author_name = message.author.name
            self.abuser_id = message.author.id

            self.summary += ("Message: " + message.content + "; Author: " + message.author.name)

            return ["I found this message:", "```" + message.author.name + ": " + message.content + "```", reply]
        
        # if multiple incorrect choices, close report
        if self.num_attempts == 3:
            reply = "You have provided an invalid option 3 times. "
            reply += "We are thus closing this report. Please feel free to start the reporting process again."
            self.state = State.REPORT_CANCELLED
            return [reply]
        
        if self.state == State.MESSAGE_IDENTIFIED:
            if message.content.strip().lower() == "impersonation":
                self.num_attempts = 0
                self.state = State.IMPERSONATION_INIT
                self.summary += ("; Abuse type: impersonation")
                reply = "You have begun the impersonation reporting flow. "
                reply += "Is the account impersonating someone? (Yes/No)"

            elif message.content.strip().lower() == "misinformation":
                self.state = State.MISINFORMATION_INIT
                self.num_attempts = 0
                reply = "You have begun the misinformation reporting flow. "
                reply += "Have you encountered misinformation? (Yes/No)"
            else:
                self.num_attempts += 1
                reply = "Invalid option %s provided. Please enter either \"misinformation\" or \"impersonation\" "
            return [reply]
        
        if self.state == State.IMPERSONATION_INIT or self.state == State.MISINFORMATION_INIT:
            if message.content.strip().lower() == "yes":
                self.num_attempts = 0
                if self.state == State.IMPERSONATION_INIT:
                    reply = "Confirm the username of the account you are reporting:"
                else:
                    reply = "What kind of misinformation have you encountered? "
                    reply += "Please enter one of the following: "
                    reply += "Manipulation, Fabrication, Misleading, Other, Satire/Parody"
                self.state = State.YES_IMPERSONATION if self.state == State.IMPERSONATION_INIT  else State.YES_MISINFORMATION
            elif message.content.strip().lower() == "no":
                self.num_attempts = 0
                self.state = State.REPORT_CANCELLED
                reply = "You have not indicated impersonation or misinformation. We are thus closing this report. "
                reply += "Please follow the reporting flow the for the relevant abuse type."
            else:
                self.num_attempts += 1
                reply = "You have entered an invalid option for the question. "
                reply += "Please enter yes or no to confirm if you are encountering online abuse."
            return [reply]
        
        if self.state == State.YES_IMPERSONATION:
            if message.content.strip() != self.author_name:
                reply = "The author name confirmed here %s does not match the author name of the reported message %s. " % (message.content.strip(), str(self.author_name))
                reply += "We are thus closing this report."
                self.state = State.REPORT_CANCELLED
            else:
                reply = "You have confirmed %s as the person impersonating someone. " % (self.author_name)
                reply += "Who is being impersonated? Enter \"1\" if it you or an organisation you represent and \"2\" if it is someone else"
                self.state = State.WHO_IMPERSONATION
            return [reply]
        
        if self.state == State.WHO_IMPERSONATION:
            parsed_msg = int(message.content.strip()) if message.content.strip().isdigit() else None
            if parsed_msg == 1:
                self.num_attempts = 0
                self.state = State.USER_IMPERSONATED
                self.summary += "; Victim of impersonation: user and/or user's organization"
                reply = "You have confirmed that you or an organisation that you represent is being impersonated. "
                reply += "Please provide a government identification number to verify your identity."
            elif parsed_msg == 2:
                self.num_attempts = 0
                self.state = State.SOMEONE_IMPERSONATED
                self.summary += "; Victim of impersonation: someone else"
                reply = "You have confirmed that someone else is being impersonated. " 
                reply += "Why do you believe the account is impersonating someone?"
            else:
                self.num_attempts += 1
                reply = "You entered an invalid option %s. Please enter 1 or 2." % (message.content.strip())
            return [reply]

        if self.state == State.USER_IMPERSONATED or self.state == State.SOMEONE_IMPERSONATED:
            self.summary += ("; Info on impersonation: " + message.content.strip())
            reply = "Thank you for providing the following information about your impersonation report: %s. " % (message.content.strip())
            reply += "Please feel free to provide any other information:"
            self.state = State.ADDITIONAL_INFO


        if self.state == State.YES_MISINFORMATION:
            if message.content.strip().lower() in ["manipulation", "fabrication", "misleading", "other", "satire", "parody"]:
                self.summary += ("; Abuse type: misinformation -- " + message.content.strip().lower())
                self.num_attempts = 0
                if message.content.strip().lower() in ["satire", "parody"]:
                    reply = "This content does not violate our community guidelines."
                    self.state = State.REPORT_CANCELLED
                else:
                    reply = "Please feel free to provide any other information: "
                    self.state = State.ADDITIONAL_INFO
            else:
                self.num_attempts += 1
                reply = "You entered an invalid misinformation type %s. Please enter the misinformation type again: " % (message.content)
            return [reply]
            

        if self.state == State.ADDITIONAL_INFO:
            self.summary += ("; Additional  info: " + message.content.strip())
            reply = "Thank you for your efforts in making our platform safer. Our team will review your report shortly!"
            self.state = State.AWAITING_REVIEW
            return [reply]


        if self.state == State.AWAITING_REVIEW:
            reply = "Your report is awaiting review by a moderator. Please wait for this report to be completed before submitting another report."
            return [reply]
            

        return []
    

    def print_moderator_summary(self):
        return [self.summary]
    

    def report_awaiting_review(self):
        return self.state == State.AWAITING_REVIEW
    

    def mark_completed(self):
        self.state = State.REPORT_COMPLETE

    def report_complete(self):
        return self.state == State.REPORT_COMPLETE or self.state == State.REPORT_CANCELLED
    


    

