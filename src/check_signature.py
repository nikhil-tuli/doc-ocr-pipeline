#from paddleocr import PPStructureV3
#help(PPStructureV3)

#from paddleocr import PPStructureV3
#print(paddleocr.__version__)
#engine = PPStructureV3()

#from __future__ import annotations
#from src.ocr_engine import PPStructureEngine
#engine = PPStructureEngine(lang="en", device="cpu")

import fitz
src = fitz.open("Swiggy-Q1-FY2027-Shareholder-letter.pdf")
subset = fitz.open()
for page_num in [3,4,5,6,7,8]:   # pick a spread across the doc
    subset.insert_pdf(src, from_page=page_num, to_page=page_num)
subset.save("swiggy_subset_test.pdf")

'''engine = PPStructureV3(lang='en')
result = engine.predict("Swiggy-Q1-FY2027-Shareholder-letter.pdf")
page_res = result[0]

for block in page_res["parsing_res_list"]:
    if block.label == "table":
        print(block.to_dict())
        break
else:
    print("No table blocks found in parsing_res_list")'''