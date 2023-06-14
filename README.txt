@author: barbosl2

Brief summary of the files.
Finished Models folder have all the ran model with extension .h5 as a repository
Dump folder have files that were used to support development or explanations, not essential to the code development
Inputs folder is for the raw data and treated iamges such as macrographs and masks
Outputs folder is to save outputs such as files after treatment, outputs of the various scripts and of the finished models. Note, may have files overlap since the inputs of some scripts is the inputs of others.

ConvertImages.py: image treatment to get the right size and properties for both the images and masks
LoadUnet.py: Loads a Unet model and get predictions on the outputs folder
StreamlitCrackApp.py: Streamlit code for model deployment. Must run with code "streamlit run StreamlitCrackApp.py"
Unet.py: Train a new U-net model with the given parameters and input files and saves it on Finished Models folder

Presentation_SCC identification using deep learning.pptx: Presentation
OnePage_SCC identification using deep learning.pdf: OnePage

.gitignore: Files to be ignored by git