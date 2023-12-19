from PIL import Image
import pytesseract

# Path to the image
img_path = '/Users/rimdimjim/Desktop/Screen Shot 2023-12-06 at 11.00.13 AM.png'

# Open the image
img = Image.open(img_path)

# Use Tesseract to perform OCR on the image
text = pytesseract.image_to_string(img)

text = text.strip()  # Remove any extra white spaces or new lines
