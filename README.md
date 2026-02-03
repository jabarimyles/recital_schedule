# 🩰 Ballet Recital Scheduler

A web application to create optimized performance schedules for dance recitals.

## Features

- **Costume change management**: Ensures dancers have enough time between performances
- **Sibling grouping**: Keeps siblings' performances on the same day
- **Balanced scheduling**: Distributes performances evenly across days
- **Age-based ordering**: Creates natural flow from younger to older performers
- **Custom constraints**: Honors same-day and different-day requirements
- **Integer Linear Programming**: Uses PuLP optimizer for mathematically optimal solutions

## 🚀 Deploy to Streamlit Community Cloud (Free!)

### Step 1: Create a GitHub Account (if you don't have one)
1. Go to [github.com](https://github.com)
2. Click "Sign Up" and create a free account

### Step 2: Create a New Repository
1. Once logged in, click the **+** icon in the top right → **New repository**
2. Name it `ballet-recital-scheduler` (or any name you like)
3. Make sure it's set to **Public**
4. Click **Create repository**

### Step 3: Upload the Files
1. On your new repository page, click **"uploading an existing file"** link
2. Drag and drop these 3 files:
   - `app.py`
   - `requirements.txt`
   - `README.md`
3. Click **Commit changes**

### Step 4: Deploy on Streamlit Cloud
1. Go to [share.streamlit.io](https://share.streamlit.io)
2. Click **"Sign in with GitHub"** and authorize Streamlit
3. Click **"New app"**
4. Fill in:
   - **Repository**: Select your `ballet-recital-scheduler` repo
   - **Branch**: `main`
   - **Main file path**: `app.py`
5. Click **"Deploy!"**

### Step 5: Wait & Share
- Deployment takes 2-5 minutes
- Once complete, you'll get a URL like: `https://your-app-name.streamlit.app`
- Share this URL with anyone who needs to create schedules!

### 💡 Tips
- The app is **free** and stays live as long as it gets occasional traffic
- If it goes to sleep after inactivity, it wakes up automatically when someone visits
- You can customize the URL in Streamlit Cloud settings

## Using the App

1. **Download the template** from the sidebar
2. **Fill in your data** in Excel:
   - **Configuration**: Number of days, start time, costume change time
   - **Classes**: All dance classes/performances
   - **Dancers**: All participating dancers
   - **Enrollments**: Which dancers are in which classes
   - **Siblings**: Group siblings together (same Sibling_Group number)
   - **Constraints**: Same-day or different-day requirements
3. **Upload your completed file**
4. **Click "Generate Schedule"**
5. **Download** your optimized schedule

## Local Development

```bash
# Install dependencies
pip install -r requirements.txt

# Run the app
streamlit run app.py
```

## File Structure

```
ballet-scheduler/
├── app.py              # Main Streamlit application
├── requirements.txt    # Python dependencies
└── README.md          # This file
```

## Data Format

### Configuration Sheet
| Parameter | Value | Description |
|-----------|-------|-------------|
| num_days | 3 | Number of recital days |
| min_change_time | 30 | Minutes for costume changes |
| start_time | 14:00 | Performance start time |
| max_duration_per_day | 240 | Max minutes per day |
| setup_buffer | 5 | Minutes between performances |

### Classes Sheet
| Class_Name | Duration_Minutes | Average_Age | Number_of_Dancers |
|------------|------------------|-------------|-------------------|
| Ballet_1 | 4 | 6 | 12 |

### Enrollments Sheet
| Dancer_ID | Class_Name |
|-----------|------------|
| 1 | Ballet_1 |
| 1 | Jazz_1 |

### Siblings Sheet
| Sibling_Group | Dancer_ID |
|---------------|-----------|
| 1 | 1 |
| 1 | 2 |

### Constraints Sheet
| Constraint_Type | Class1 | Class2 | Reason |
|-----------------|--------|--------|--------|
| same_day | Ballet_1 | Ballet_2 | Share students |

## License

MIT License - Feel free to use and modify!
