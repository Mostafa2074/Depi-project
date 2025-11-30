# [file name]: email_alert.py
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
import os
import re
import streamlit as st
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
        # Load environment variables from .env file
        load_dotenv(env_path)
        
        # Try Streamlit secrets first (for cloud deployment)
        try:
            self.__sender_email = st.secrets.get("SENDER_EMAIL", "")
            self.__sender_password = st.secrets.get("SENDER_PASSWORD", "")
            if self.__sender_email and self.__sender_password:
                print("✅ Using email credentials from Streamlit secrets")
        except:
            self.__sender_email = ""
            self.__sender_password = ""
        
        # Fallback to .env file if secrets not available
        if not self.__sender_email or not self.__sender_password:
            self.__sender_email = os.getenv("SENDER_EMAIL")
            self.__sender_password = os.getenv("SENDER_PASSWORD")
            if self.__sender_email and self.__sender_password:
                print("✅ Using email credentials from .env file")
        
        # Debug information
        print(f"Sender email configured: {'Yes' if self.__sender_email else 'No'}")
        
        # sender's email and password validation
        if not self.__sender_email or not self.__sender_password:
            raise ValueError(
                "Sender email and password must be set in .env file or Streamlit secrets.\n"
                "Please make sure your .env file contains:\n"
                "SENDER_EMAIL=your_email@gmail.com\n"
                "SENDER_PASSWORD=your_app_password\n\n"
                "Or set them in Streamlit secrets as SENDER_EMAIL and SENDER_PASSWORD"
            )
        
        if not isinstance(self.__sender_email, str) or not isinstance(self.__sender_password, str):
            raise ValueError("Sender email and password must be strings.")
        
        if len(self.__sender_email) == 0 or len(self.__sender_password) == 0:
            raise ValueError("Sender email and password cannot be empty.")
        
        if re.match(r"[^@]+@[^@]+\.[^@]+", self.__sender_email) is None:
            raise ValueError("Invalid sender email address format.")
    
        
        self.__stmp_server = "smtp.gmail.com"
        self.__smtp_port = 587

        # Initialize server connection (will be established when needed)
        self.__server = None

        print("EmailAlert initialized successfully.")

    # static method to get singleton instance
    @staticmethod
    def get_instance(env_path=".env"):
        if EmailAlert._instance is None:
            EmailAlert._instance = EmailAlert(env_path)
        return EmailAlert._instance

    def _ensure_connection(self):
        """Ensure SMTP connection is established"""
        if self.__server is None:
            try:
                print("Connecting to SMTP server...")
                self.__server = smtplib.SMTP(self.__stmp_server, self.__smtp_port)
                self.__server.starttls()  # Secure connection
                print("Logging in to email server...")
                self.__server.login(self.__sender_email, self.__sender_password)
                print("Email server connected successfully.")
            except Exception as e:
                print(f"Failed to connect to email server: {e}")
                # More detailed error information
                if "Authentication failed" in str(e):
                    print("Authentication failed. Please check your email and password.")
                    print("Note: For Gmail, you need to use an App Password, not your regular password.")
                elif "Connection refused" in str(e):
                    print("Connection refused. Please check your network connection and SMTP settings.")
                raise

    def send_email(self, email_content_instance):
        # Ensure connection is established
        self._ensure_connection()
        
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

        try:
            print(f"Sending email to {email_content_instance.recipient}...")
            self.__server.sendmail(self.__sender_email, email_content_instance.recipient, message.as_string())
            print(f"Email sent to {email_content_instance.recipient} successfully.")
            return True
        except Exception as e:
            print(f"Failed to send email: {e}")
            # Re-establish connection on failure
            self.__server = None
            self._ensure_connection()
            raise

    def __del__(self):
        """Clean up SMTP connection"""
        if self.__server:
            try:
                self.__server.quit()
                print("SMTP connection closed.")
            except:
                pass
