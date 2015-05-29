from char_processing import replace_special_char
from char_processing import strip_accents


def text_normalization_and_tokenization( text , joinTokens=False , specialChars=None ):
    if specialChars is None:
        specialChars = set()
    text = unicode( text )
    # special chars
    # remove diatrics
    # to lower
    text = strip_accents( replace_special_char( unicode( text ) ) ).lower()
    # keep az09
    flagToken = False
    flagAlpha = False
    start = 0
    tokens = []
    tokens_append = tokens.append
    for i , c in enumerate( text ):
        if c in specialChars:
            if flagToken:
                flagToken = False
                tokens_append( text[start:i] )
            tokens_append( c )
        elif 'a' <= c <= 'z':
            if not flagToken :
                flagToken = True
                flagAlpha = True
                start = i
            elif not flagAlpha:
                #tokens_append( text[start:i])
                tokens_append( "<NUM>" )
                start = i
                flagAlpha = True
        elif '0' <= c <= '9':
            if not flagToken:
                flagToken = True
                flagAlpha = False
                flagNum = True
                start = i
            elif flagAlpha:
                tokens_append( text[start:i])
                start = i
                flagAlpha = False
        elif flagToken:
            flagToken = False
            if not flagAlpha:
                tokens_append( "<NUM>" )
            else :
                tokens_append( text[start:i] )
    if flagToken:
        if not flagAlpha:
            tokens_append( "<NUM>" )
        else :
            tokens_append( text[start:i+1] )
    if joinTokens:
        return ' '.join( tokens )
    else:
        return tokens
