import re

AGE_REPLAC = "_AGE_"
ANON_REPLAC = "_ANON_"
CODE_REPLAC = "_CODE_"
MEASURE_REPLAC = "_MEAS_"
NUMBER_REPLAC = "_NUM_"


def minimal_cleaning(s):
    s = ' '.join(t.lower() for t in s.split())
    s = s.replace("..", ".")
    s = s.replace(",", " ")
    s = s.replace(";", " ")
    s = remove_list_items(s)
    s = substitute_age(s)
    s = substitute_code(s)
    s = substitute_anon(s)
    s = substitute_measures(s)
    s = substitute_numbers(s)
    s = substitute_multiple_anon(s)
    if not s.endswith("."):
        s = s + "."
    return s


def apply_pattern(pattern, s, replacement=''):
    s = re.sub(pattern, replacement, s)
    return s.strip()


def remove_list_items(s):
    # pattern reads: \d single digit, \. dot, \s*  white chars, (?= [a..Z]) positive lookahead single letters
    pattern = r'\d\.\s*(?=[a-zA-Z])'
    return apply_pattern(pattern, s)


def substitute_anon(s):
    pattern = r'x{3,}(-[a-zA-Z]+)?'
    return apply_pattern(pattern, s, ANON_REPLAC)


def substitute_age(s):
    pattern = r'x{2,}[-|\s](year[-|\s]old|old|yo)'
    out = apply_pattern(pattern, s, AGE_REPLAC)
    return out


def substitute_code(s):
    pattern = r'(x)+-(x)+'
    return apply_pattern(pattern, s, CODE_REPLAC)


def substitute_measures(s):
    pattern = r'\d{,2}\.\d{,2}\s*(mm|cm|in)'
    return apply_pattern(pattern, s, MEASURE_REPLAC)


def substitute_numbers(s):
    pattern = r'\d+\.?\d*'
    return apply_pattern(pattern, s, NUMBER_REPLAC)


def substitute_multiple_anon(s):
    # pattern = r'@ANON[\s*@ANON]+'
    # out = apply_pattern(pattern, s, ANON_REPLAC)
    # pattern = r'@CODE[\s*@CODE]+'
    # out = apply_pattern(pattern, s, CODE_REPLAC)
    pattern = r"(@ANON)(\s\1)+"
    out = apply_pattern(pattern, s, "@ANON")
    # print('\t%s ----> %s' % (s, out))
    return out

# ---
def test():
    a = 'aAa 1.'
    print("Input: |%s|" % a)
    print("Minimal, output: |%s|" % minimal_cleaning(a))

    a = '1.x 2. y'
    print("Input: |%s|" % a)
    print("Minimal, output: |%s|" % minimal_cleaning(a))

    a = 'xxxx-x xxx-x xxx-a'
    print("Input: |%s|" % a)
    print("Minimal, output: |%s|" % minimal_cleaning(a))

    a = 'xxxx-x xxx-x xxx-a 1.2 cm 12131.23131 a'
    print("Input: |%s|" % a)
    print("Minimal, output: |%s|" % minimal_cleaning(a))

    a = "Comparison XXXX, XXXX. Well-expanded and clear lungs. Mediastinal contour within normal limits. No acute cardiopulmonary abnormality identified."
    print("Input: |%s|" % a)
    print("Minimal, output: |%s|" % minimal_cleaning(a))

    a = "1. There is abnormal separation of right XXXX XXXX, question very acute versus chronic injury. Correlate for focal pain. If indicated consider dedicated right shoulder films. 2. No acute cortical artery disease."
    print("Input: |%s|" % a)
    print("Minimal, output: |%s|" % minimal_cleaning(a))
