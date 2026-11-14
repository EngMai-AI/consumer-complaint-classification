clean_text(complaint), truncation=True, max_length=256))

Input text should be cleaned the same way as the training data.

Run the notebook
bash
pip install -r requirements.txt
jupyter notebook Consumer_Complaint_Classification.ipynb

DistilBERT fine-tuning was done on a GPU in Google Colab. Training on CPU will be very slow.

Repository structure
Consumer_Complaint_Classification.ipynb   data exploration, training, evaluation, comparison
requirements.txt                          dependencies
README.md
LICENSE
Limitations
Categories overlap, mainly credit reporting vs. debt collection
English text from a single source (US consumer complaints); performance on other domains or languages is not guaranteed
Only 5 coarse product categories; it does not predict issue type or sub-product
Meant to assist routing, not to replace human review
License

MIT
