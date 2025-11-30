import re
import os.path as path
class EmailContent:
    def __init__(self , html_template_path = "email_alert.html"):
        self.__html_template_path = html_template_path
        if path.exists(self.__html_template_path):
            self.__html_content = open(self.__html_template_path, "r", encoding="utf-8").read()
        else:
            raise FileNotFoundError(f"HTML template file not found at {self.__html_template_path}")
   


    @property
    def html_content(self):
        return self.__html_content

    @property
    def recipient(self):
        return self.__recipient
    @recipient.setter
    def recipient(self, value):
        if not isinstance(value, str):
            raise ValueError("Recipient email must be a string.")
        if re.match(r"[^@]+@[^@]+\.[^@]+", value) is None:
            raise ValueError("Invalid email address format.")
        self.__recipient = value

    @property
    def subject(self):
        return self.__subject
    @subject.setter
    def subject(self, value):
        if not isinstance(value, str):
            raise ValueError("Subject must be a string.")
        if len(value) == 0:
            raise ValueError("Subject cannot be empty.")
        if len(value) > 150:
            raise ValueError("Subject length must not exceed 150 characters.")
        self.__subject = value

    @property
    def message_body(self):
        return self.__message_body
    @message_body.setter
    def message_body(self, value):
        bodyKeys = {'title' : str, 'message':str, 'table_rows': dict, 'notes': str}
        if not isinstance(value, dict):
            raise ValueError("Message body must be a dictionary.")
        if not all(key in value.keys() for key in bodyKeys.keys()):
            raise ValueError(f"Message body missing the following keys: {', '.join(set(bodyKeys.keys()) - set(value.keys()))}")
        for key in bodyKeys.keys():
            if not isinstance(value[key], bodyKeys[key]):
                raise ValueError(f"{key} in message body must be of type {bodyKeys[key].__name__}.")
        self.__message_body = value
    
    def prepare_html(self):
        if self.recipient is None or self.subject is None or self.message_body is None:
            raise ValueError("Recipient, subject, and message body must be set in email content.")



        if self.__html_content and self.message_body:
            html_content = self.__html_content
            html_content = html_content.replace("{{title}}", self.message_body['title'])
            html_content = html_content.replace("{{message}}", self.message_body['message'])
            html_content = html_content.replace("{{notes}}", self.message_body['notes'])
            table_rows_str = ""
            for row_key, row_value in self.message_body['table_rows'].items():
                table_rows_str += f"""
                <tr>
                    <td>
                      <strong>{row_key}: </strong>
                    </td>
                    <td>{row_value}</td>
                  </tr> 
                """
            html_content = html_content.replace("{{table_rows}}", table_rows_str)
            self.__html_content =  html_content
        if self.__html_content is None:
            raise ValueError("HTML content is not loaded in EmailContent instance.")