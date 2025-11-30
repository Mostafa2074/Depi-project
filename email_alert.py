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
    

    class EmailAlert:
    # ... existing code ...
    
    def __init__(self, env_path=".env"):
        # Use Streamlit secrets as primary source
        try:
            self.__sender_email = st.secrets.get("EMAIL", {}).get("SENDER_EMAIL", "")
            self.__sender_password = st.secrets.get("EMAIL", {}).get("SENDER_PASSWORD", "")
            
            # Fallback to environment variables ONLY if secrets not available
            if not self.__sender_email or not self.__sender_password:
                st.warning("⚠️ Using environment variables as fallback")
                self.__sender_email = os.getenv("SENDER_EMAIL", "")
                self.__sender_password = os.getenv("SENDER_PASSWORD", "")
                
        except Exception as e:
            st.error(f"❌ Error loading email configuration: {e}")
            # Fallback to environment variables
            self.__sender_email = os.getenv("SENDER_EMAIL", "")
            self.__sender_password = os.getenv("SENDER_PASSWORD", "")

        # Validation
        if not self.__sender_email or not self.__sender_password:
            st.error("❌ Email credentials not found. Please configure .streamlit/secrets.toml")
            raise ValueError("Email credentials not configured")
            
        if not re.match(r"[^@]+@[^@]+\.[^@]+", self.__sender_email):
            raise ValueError("Invalid sender email address format")

        self.__stmp_server = "smtp.gmail.com"
        self.__smtp_port = 587
        self.__server = None
        
        st.success("✅ Email alert system initialized successfully!")

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

