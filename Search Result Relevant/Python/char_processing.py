# -*- coding: utf-8 -*-

def remove_white_space( key ):
    return ''.join( key.split() )

#remove non AN char
ps = list( set( [ '_' , '\\' , '=' , '.' , '+' , '-' , "'" , '?' , '!' , '#' , '"' , '$' , '(' , ')' , '*' , '/' , '^' , '~' , '`' , '&' , ':' , ';' , ',' , u'µ' , u'’' , u'£' , u'§' , u'©' , u'¨' , u'«' , u'°' , u'»' , u'½' , u'´' , u'¸' , u'²' , u'×' , u'º' , u'€' , u'–' , u'•' , u'Œ' , u'×', u'ø' , u'|' , u'®' ] ) )
def strip_punc( s ):
    for p in ps:
        s = s.replace( p , ' ' )
    return s

#replace diatrics
def strip_accents( s ):
    import unicodedata
    return ''.join((c for c in unicodedata.normalize('NFD', s) if unicodedata.category(c) != 'Mn'))

#replace spécial char
specialCharMap = {u'œ':"oe", u'æ':"ae", u'ß':"ss"}
def replace_special_char(inputText):
	outputText = inputText
	for specialChar, replacement in specialCharMap.items():
		outputText = outputText.replace(specialChar, replacement)
	return outputText


#remoce extra space
def strip_duplicated_space( s ):
    return ' '.join( s.replace( '\t' , ' ' ).replace( '\n' , ' ' ).split() )

#filter AN String
def keep_az09_chars( s , keepExtraChars = '' ):
    return ''.join( c for c in s if ( 'a' <= c <= 'z' ) or ( '0' <= c <= '9' ) or c == ' ' or c in keepExtraChars )


def keep_az_chars( s , keepExtraChars = '' ):
    return ''.join( c for c in s if ( 'a' <= c <= 'z' ) or c == ' ' or c in keepExtraChars )


