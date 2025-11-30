# [file name]: email_alert.py
# [file content begin]
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart

import os
import re
from dotenv import load_dotenv

# Remove relative import and use direct import instead
try:
    from email_content import EmailContent
except ImportError:
    # Fallback for different directory structures
    try:
        from .email_content import EmailContent
    except ImportError:
        EmailContent = None
        print("Warning: EmailContent module not found")


class EmailAlert:
    # singleton instance
    _instance = None

    def __new__(cls, env_path=".env"):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance
    

    def __init__(self, env_path=".env"):
        # load environment variables from .env file
        load_dotenv(env_path)

        self.__sender_email = os.getenv("SENDER_EMAIL")
        self.__sender_password = os.getenv("SENDER_PASSWORD")

        # sender's email and password validation
        if not self.__sender_email or not self.__sender_password:
            raise ValueError("Sender email and password must be set in environment variables.")
        if not isinstance(self.__sender_email, str) or not isinstance(self.__sender_password, str):
            raise ValueError("Sender email and password must be strings.")
        if len(self.__sender_email) == 0 or len(self.__sender_password) == 0:
            raise ValueError("Sender email and password cannot be empty.")
        if re.match(r"[^@]+@[^@]+\.[^@]+", self.__sender_email) is None:
            raise ValueError("Invalid sender email address format.")
    
        
        self.__stmp_server = "smtp.gmail.com"
        self.__smtp_port = 587

        # start connection to SMTP server
        self.__server = smtplib.SMTP(self.__stmp_server, self.__smtp_port)
        self.__server.starttls()                     # Secure connection
        self.__server.login(self.__sender_email, self.__sender_password)

        print("Email server connected successfully.")

    # static method to get singleton instance
    @staticmethod
    def get_instance(env_path=".env"):
        if EmailAlert._instance is None:
            EmailAlert._instance = EmailAlert(env_path)
        return EmailAlert._instance

    def send_email(self, email_content_instance):
        if not self.__server:
            raise ConnectionError("SMTP server is not connected.")
        
        # Check if EmailContent class is available
        if EmailContent is None:
            raise ImportError("EmailContent module is not available.")
        
        if not isinstance(email_content_instance, EmailContent):
            raise ValueError("email_content must be an instance of EmailContent class.")
        
        if not email_content_instance.recipient or not email_content_instance.subject or not email_content_instance.message_body:  
            raise ValueError("Recipient, subject, and message body must be set in email content.")
        

        if email_content_instance.html_content is None:
            raise ValueError("HTML content is not loaded in EmailContent instance.")
        
        # Email message content
        message = MIMEMultipart()
        message["From"] = self.__sender_email

        message["To"] = email_content_instance.recipient
        message["Subject"] = email_content_instance.subject
        message.attach(MIMEText(email_content_instance.html_content, "html"))

        self.__server.sendmail(self.__sender_email, email_content_instance.recipient, message.as_string())
        print(f"Email sent to {email_content_instance.recipient} successfully.")
# [file content end]