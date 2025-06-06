import logging
from tkinter import messagebox, simpledialog
import threading
import tkinter as tk
from difflib import get_close_matches

thread_lock = threading.Lock()

__all__ = ['log_', 'ThrowDlg', 'match_keywords']

logging.basicConfig(format='%(asctime)s | %(name)s | %(levelname)s | %(message)s\n', level=logging.INFO)

class Color:
    def __init__(self, name, normal, bold, italic):
        self.name = name
        self.normal = normal
        self.bold = bold
        self.italic = italic

    def __str__(self):
        return self.normal
    
class Colors:
    def __init__(self):
        colors = {
            'BLUE': ('\033[34m', '\033[1;34m', '\033[3;34m'),
            'GREEN': ('\033[32m', '\033[1;32m', '\033[3;32m'),
            'RED': ('\033[31m', '\033[1;31m', '\033[3;31m'),
            'YELLOW': ('\033[33m', '\033[1;33m', '\033[3;33m'),
            'PURPLE': ('\033[35m', '\033[1;35m', '\033[3;35m'),
            'CYAN': ('\033[36m', '\033[1;36m', '\033[3;36m'),
            'MAGENTA': ('\033[35m', '\033[1;35m', '\033[3;35m'),
            'WHITE': ('\033[37m', '\033[1;37m', '\033[3;37m'),
            'BLACK': ('\033[30m', '\033[1;30m', '\033[3;30m'),
            'DARK_GREY': ('\033[90m', '\033[1;90m', '\033[3;90m'),
            'LIGHT_GREY': ('\033[37m', '\033[1;37m', '\033[3;37m'),
            'LIGHT_RED': ('\033[91m', '\033[1;91m', '\033[3;91m'),
            'LIGHT_GREEN': ('\033[92m', '\033[1;92m', '\033[3;92m'),
            'LIGHT_YELLOW': ('\033[93m', '\033[1;93m', '\033[3;93m'),
            'LIGHT_BLUE': ('\033[94m', '\033[1;94m', '\033[3;94m'),
            'LIGHT_PURPLE': ('\033[95m', '\033[1;95m', '\033[3;95m'),
            'LIGHT_CYAN': ('\033[96m', '\033[1;96m', '\033[3;96m'),
            'LIGHT_MAGENTA': ('\033[95m', '\033[1;95m', '\033[3;95m'),
            'DEFAULT': ('\033[0m', '\033[1;0m', '\033[3;0m'),
        }
        
        for color_name, (normal, bold, italic) in colors.items():
            setattr(self, color_name.lower(), Color(color_name, normal, bold, italic))
            
        self.reset = '\033[0m'
        
    def __str__(self):
        return self.reset

colors = Colors()

def match_keywords(word: str, keywords: list, threshold: float = 0.8) -> str:
    """
    Match a given word to the closest keyword from a provided list of keywords.

    Parameters
    ----------
    word : str
        The word that needs to be matched.
    keywords : list
        A list of potential keyword matches.

    Returns
    -------
    word : str
        The best-matching keyword from the list based on string similarity.

    Description
    -----------
    This function tries to find the most similar keyword to the input word by checking if the input word or any part of it matches with a keyword.
    If no exact match is found, it calculates similarity scores based on the intersection of letters in the word and keywords, choosing the closest match.

    - If the word partially matches a keyword, the function logs a warning and returns the matching keyword.
    - If no direct match is found, a similarity score is calculated by comparing the characters of the word and keywords, and the closest match is returned.

    Logging
    -------
    The function logs both partial matches and the final match with a similarity score using the `log_` function.

    Example
    -------
    ```python
    keywords = ['linear', 'polynomial', 'neural_network']
    match = match_keywords('line', keywords)
    # Output: 'linear'
    ```
    """

    for keyword in keywords:
        if keyword in word or word in keyword:
            # log_(f'"{word}" matched with "{keyword}" directly.', color='green', level='info')
            return keyword

    # Use difflib to find the closest match based on similarity
    matches = get_close_matches(word, keywords, n=1, cutoff=threshold)
    if matches:
        closest_match = matches[0]
        log_(f'"{word}" ~= "{closest_match}", score >= {threshold}', color='green', level='info', font_style='italic')
        return closest_match
    else:
        # log_(f'No close match for "{word}" found. Returning original.', color='red', level='warning')
        return word

    
def log_(message: str, color: str='default', font_style: str='normal', level: str='info'):
    """
    Log a message with customizable color, font style, and severity level.

    Parameters
    ----------
    message : str
        The message to log.
    color : str, optional
        The color of the log message (default is 'default'). Must be one of:
        ['blue', 'green', 'red', 'yellow', 'purple', 'cyan', 'magenta', 'white', 'black', 'default',
         'dark_grey', 'light_grey', 'light_red', 'light_green', 'light_yellow', 'light_blue',
         'light_purple', 'light_cyan', 'light_magenta', 'reset'].
    font_style : str, optional
        The font style of the log message (default is 'normal'). Must be one of:
        ['normal', 'bold', 'italic'].
    level : str, optional
        The severity level of the log (default is 'info'). Must be one of:
        ['info', 'warning', 'error', 'critical'].

    Description
    -----------
    This function allows logging messages with different color schemes and font styles. 
    It validates the color, font style, and logging level inputs using the `match_keywords` function 
    to find the closest match from predefined options.

    Example
    -------
    ```python
    log_("Sample message", color="blue", font_style="bold", level="info")
    ```
    """
    global colors
    color_must_be = ['blue', 'green', 'red', 'yellow', 'purple', 'cyan', 'magenta', 'white', 'black', 'default'
                     'dark_grey', 'light_grey', 'light_red', 'light_green', 'light_yellow', 'light_blue',
                     'light_purple', 'light_cyan', 'light_magenta', 'reset']
    
    font_must_be = ['normal', 'bold', 'italic']
    level_must_be = ['info', 'warning', 'error', 'critical']

    with thread_lock:
        if color.lower() not in color_must_be:
            color = match_keywords(color, color_must_be)

        if font_style.lower() not in font_must_be:
            font_style = match_keywords(font_style, font_must_be)

        if level.lower() not in level_must_be:
            level = match_keywords(level, level_must_be)

        color_attr = getattr(colors, color.lower(), colors.default)
        font_style_attr = getattr(color_attr, font_style.lower(), color_attr.normal)
        log_func = getattr(logging, level.lower(), logging.info)
        
        msg = f'{font_style_attr}{message}{colors.reset}'
        log_func(msg)


class ThrowDlg:

    """
    A utility class for displaying various dialog boxes (warnings, errors, info, and prompts) using Tkinter.

    Methods
    -------
    warn(msg: str) -> str
        Displays a warning message box with the specified message and returns the user's response.
        
    error(msg: str) -> str
        Displays an error message box with the specified message and returns the user's response.
        
    info(msg: str) -> str
        Displays an information message box with the specified message and returns the user's response.
        
    yesno(msg: str) -> str
        Displays a Yes/No question dialog box with the specified message and returns "yes" or "no" based on the user's choice.
        
    okcancel(msg: str) -> str
        Displays an OK/Cancel question dialog box with the specified message and returns "ok" or "cancel" based on the user's choice.
        
    input(msg: str) -> str
        Displays an input prompt dialog box asking for a string response from the user.

    Description
    -----------
    - This class is a wrapper around the Tkinter library's message box and input dialog functionality, providing an easy way to display dialogs in applications.
    - Each method initializes a root Tkinter window (hidden from the user), invokes the appropriate dialog box, and then closes the root window.
    - The class is designed to simplify user interactions such as showing alerts, confirming actions, or gathering input.

    Example
    -------
    ```python
    from dialog_class import ThrowDlg

    response = ThrowDlg.yesno("Do you want to proceed?")
    if response == "yes":
        print("Proceeding...")
    else:
        print("Operation cancelled.")
    ```
    """

    @classmethod
    def _init_root(cls):
        root = tk.Tk()
        root.withdraw()
        return root

    @classmethod
    def warn(cls, msg:str)->str:
        root = cls._init_root()
        reponse = messagebox.showwarning('Warning', msg)
        root.destroy()
        return reponse

    @classmethod
    def error(cls, msg:str)->str:
        root = cls._init_root()
        reponse = messagebox.showerror('Error', msg)
        root.destroy()
        return reponse
    
    @classmethod
    def info(cls, msg:str)->str:
        root = cls._init_root()
        reponse = messagebox.showinfo('Info', msg)
        root.destroy()
        return reponse
    
    @classmethod
    def yesno(cls, msg:str)->str:
        root = cls._init_root()
        reponse = messagebox.askyesno('Question', msg)
        root.destroy()
        return "yes" if reponse else "no"
    
    @classmethod
    def okcancel(cls, msg:str)->str:
        root = cls._init_root()
        reponse = messagebox.askokcancel('Question', msg)
        root.destroy()
        return "ok" if reponse else "cancel"
    
    @classmethod
    def input(cls, msg:str)->str:
        root = cls._init_root()
        reponse = simpledialog.askstring('Question', msg)
        root.destroy()
        return reponse
