import re

random_text_with_phone_numbers = "+1 123-456-7890 123-456-7890 (123) 987-6543 123 456 7890 123.456.7890 1234567890 12345678901 123456789012 +8 123-456-7890"
phone_numbers = re.findall(r'\+?[1-9]\d{0,2}\s?\(?\d{3}\)?[\s.-]?\d{3}[\s.-]?\d{4}', random_text_with_phone_numbers)
print(phone_numbers)    