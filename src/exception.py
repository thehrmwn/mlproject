import sys
#from src.logger import logging

def error_message_detail(error, error_detail:sys):
    """
    Handle error by giving a custom message ....
    Args:
        error (_type_): _description_
        error_detail (sys): _description_
    """
    _, _, exc_tb = error_detail.exc_info() #give the detail info of exception
    file_name = exc_tb.tb_frame.f_code.co_filename #get the filename from exception
    
    error_message = "Error occured in Python Script name [{0}], line number [{1}], error message [{2}]".format(
        file_name, exc_tb.tb_lineno, str(error)
    ) # The detail message of error
    
    return error_message

class CustomException(Exception):
    def __init__(self, error_message, error_detail:sys):
        super().__init__(error_message)
        self.error_message = error_message_detail(error_message, error_detail=error_detail)
    
    # print the error message when the error raised
    def __str__(self) -> str:
        return self.error_message
    

# Testing Exception
# if __name__ == "__main__":
#     try:
#         x = 1/0
#     except Exception as e:
#         logging.info("Logging Has Started")
#         raise CustomException(e, sys)
        