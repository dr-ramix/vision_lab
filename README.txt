Vision Lab (FER)

This repo is a research-friendly PyTorch project for Facial Expression Recognition (FER).
The current focus is on developing model architectures and pre-processing pipelines.
Training, evaluation, and inference are generic and reusable.


>> git clone https://github.com/dr-ramix/vision_lab.git
>> cd vision_lab
>> python -m venv venv
>> source venv/bin/activate   # Windows: venv\Scripts\Activate.ps1
>> pip install -r requirements.txt

Install the package:
>> cd main
>> pip install -e .

Verify:
>> python -c "import fer; print('OK')"


test: 
>> python3 scripts/train_dummy.py