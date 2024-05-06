# What is this?



[You can see this dashboard in action here!](https://wealth-advisor-dashboard-zxeuck5wssugu23mf7cjon.streamlit.app/)

## How to 

### If you want to get this app working on your computer so you can use it, play around with it, or modify it, you need:
1. A working python / Anaconda installation
1. Git 

Then, open a terminal and run these commands one at a time:

```sh
# download files (you can do this via github desktop too)
cd <path to your FIN377 folder> # make sure the cd isn't a repo or inside a repo!
git clone https://github.com/donbowen/portfolio-frontier-streamlit-dashboard.git

# move the terminal to the new folder (adjust next line if necessary)
cd portfolio-frontier-streamlit-dashboard  

# this deletes the .git subfolder, so you can make this your own repo
# MAKE SURE THE cd IS THE portfolio-frontier-streamlit-dashboard FOLDER FIRST!
rm -r -fo .git 

# set up the packages you need for this app to work 
# (YOU CAN SKIP THESE if you have already streamlit-env, or you can 
# give this one a slightly diff name by modifying the environment.yml file)
conda env create -f streamlit_env.yml
conda activate streamlit-env

# start the app in a browser window
streamlit run app.py

# open any IDE you want to modify app - spyder > jupyterlab for this
spyder  # and when you save the file, the app website will update
```

### To deploy the site on the web, 
1. Use Github Desktop to make this a repo your own account. 
1. Go to streamlit's website, sign up, and deploy it by giving it the URL to your repo.
1. Wait a short time... and voila!

## Update requests 

1. Easy for me: Add Github action to run `update_data_cache.py` once a month.
1. Easy for anyone: The requirements file has no version restrictions. We should set exact versions.

## Notes

While it seems duplicative to have a `requirements.txt` and a  `streamlit_env.yml`, the former is needed by Streamlit and the latter makes setting up a conda environment quickly easy. So keep both. 
