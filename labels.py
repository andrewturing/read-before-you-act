"""Look-alike label pairs shared by the scene renderer and the reading test."""

# Look-alike generic drug names (confusable pairs of the kind listed in medication-safety guidance).
NAME_PAIRS = [
    ("Hydroxyzine", "Hydralazine"), ("Cefixime", "Cefuroxime"), ("Amlodipine", "Amiloride"),
    ("Metformin", "Metronidazole"), ("Prednisone", "Prednisolone"), ("Clonidine", "Clonazepam"),
    ("Losartan", "Valsartan"), ("Lamotrigine", "Lamivudine"), ("Tramadol", "Trazodone"),
    ("Bupropion", "Buspirone"), ("Chlorpromazine", "Chlorpropamide"), ("Vinblastine", "Vincristine"),
    ("Dopamine", "Dobutamine"), ("Ephedrine", "Epinephrine"), ("Glipizide", "Glyburide"),
    ("Hydrocodone", "Oxycodone"), ("Methotrexate", "Metolazone"), ("Quinine", "Quinidine"),
    ("Risperidone", "Ropinirole"), ("Sulfasalazine", "Sulfadiazine"), ("Tizanidine", "Tiagabine"),
    ("Carbamazepine", "Oxcarbazepine"), ("Azithromycin", "Erythromycin"), ("Clomiphene", "Clomipramine"),
    ("Daunorubicin", "Doxorubicin"), ("Fluoxetine", "Paroxetine"), ("Guaifenesin", "Guanfacine"),
    ("Nifedipine", "Nicardipine"), ("Propranolol", "Propafenone"), ("Levofloxacin", "Levetiracetam"),
    ("Omeprazole", "Esomeprazole"), ("Captopril", "Carvedilol"),
]
DOSES = ["5 mg", "10 mg", "25 mg", "50 mg", "100 mg", "250 mg", "500 mg"]
DOSE_PAIRS = [("5 mg", "50 mg"), ("10 mg", "100 mg"), ("25 mg", "250 mg"), ("50 mg", "500 mg"),
              ("0.5 mg", "5 mg"), ("2.5 mg", "25 mg"), ("15 mg", "75 mg"), ("20 mg", "200 mg")]
DOSE_NAMES = ["Amlodipine", "Prednisone", "Losartan", "Metformin", "Warfarin", "Digoxin", "Morphine", "Furosemide"]
VI_PAIRS = [
    ("Uống 1 viên sau ăn", "Uống 2 viên sau ăn"), ("Ngày 2 lần", "Ngày 3 lần"),
    ("Uống trước ăn", "Uống sau ăn"), ("Sáng 1 viên", "Tối 1 viên"),
    ("Uống nửa viên", "Uống một viên"), ("Dùng 5 ngày", "Dùng 7 ngày"),
    ("Nhỏ mắt trái", "Nhỏ mắt phải"), ("Bôi ngoài da", "Bôi ngoài tai"),
]
ZH_PAIRS = [
    ("每日一次", "每日三次"), ("饭前服用", "饭后服用"), ("每次一片", "每次二片"),
    ("早晨服用", "睡前服用"), ("外用药", "内服药"), ("左眼滴用", "右眼滴用"),
]
