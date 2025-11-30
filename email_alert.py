# [file name]: email_alert.py
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
import os
import re
import streamlit as st

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
        # Load environment variables from .env file first
        self._load_env_file(env_path)
        
        # Try environment variables first (from .env file)
        self.__sender_email = os.getenv("SENDER_EMAIL")
        self.__sender_password = os.getenv("SENDER_PASSWORD")
        
        # Fallback to Streamlit secrets only if environment variables are not set
        if not self.__sender_email or not self.__sender_password:
            try:
                self.__sender_email = st.secrets.get("EMAIL", {}).get("SENDER_EMAIL", "")
                self.__sender_password = st.secrets.get("EMAIL", {}).get("SENDER_PASSWORD", "")
            except Exception:
                # Streamlit secrets not available
                pass

        # sender's email and password validation
        if not self.__sender_email or not self.__sender_password:
            raise ValueError("Sender email and password must be set in environment variables (.env file) or Streamlit secrets.")
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
        print(f"Using email: {self.__sender_email}")

    def _load_env_file(self, env_path):
        """Load environment variables from .env file"""
        try:
            if os.path.exists(env_path):
                print(f"Loading environment variables from {env_path}")
                with open(env_path, 'r') as f:
                    for line in f:
                        line = line.strip()
                        if line and not line.startswith('#') and '=' in line:
                            key, value = line.split('=', 1)
                            key = key.strip()
                            value = value.strip()
                            # Remove quotes if present
                            if (value.startswith('"') and value.endswith('"')) or (value.startswith("'") and value.endswith("'")):
                                value = value[1:-1]
                            os.environ[key] = value
                            print(f"Loaded: {key}={value[:4]}...")  # Show first 4 chars for security
        except Exception as e:
            print(f"Warning: Could not load .env file: {e}")

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
                self.__server = smtplib.SMTP(self.__stmp_server, self.__smtp_port)
                self.__server.starttls()  # Secure connection
                self.__server.login(self.__sender_email, self.__sender_password)
                print("Email server connected successfully.")
            except Exception as e:
                print(f"Failed to connect to email server: {e}")
                # More detailed error information
                if "Authentication failed" in str(e):
                    print("Authentication failed. Please check your email and password.")
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
            except:
                pass
