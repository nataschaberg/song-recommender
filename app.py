import streamlit as st
import pandas as pd
import plotly.express as px
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import spotipy
from spotipy.oauth2 import SpotifyClientCredentials
import random

##############################
####        SETUP         ####
##############################

SPOTIPY_CLIENT_ID = st.secrets["secret_client_id"]
SPOTIPY_CLIENT_SECRET = st.secrets["spotipy_client_secret"]
AUDIO_FEATURES = ['danceability', 'energy', 'loudness', 'speechiness', 'acousticness', 'instrumentalness', 'liveness', 'valence', 'tempo']

auth_manager = SpotifyClientCredentials(client_id=SPOTIPY_CLIENT_ID, client_secret=SPOTIPY_CLIENT_SECRET)
sp = spotipy.Spotify(auth_manager=auth_manager)

st.set_page_config(layout="wide")
pic_1 = 'https://images.pexels.com/photos/1037999/pexels-photo-1037999.jpeg?auto=compress&cs=tinysrgb&w=1260&h=750&dpr=2'
pic_3 = 'https://img.freepik.com/free-photo/headphones-mouse-orange-background_23-2148182104.jpg?w=2000&t=st=1678234576~exp=1678235176~hmac=0eefabf71f6411b037f71cc70153796c2e31c906efa4cba24b8235b74c1ea75b'
pic_2 = 'https://t3.ftcdn.net/jpg/04/14/62/04/360_F_414620426_iam5N6Bhp8YMOEOJD7MqdI1dC0Tw0CIY.jpg'

bg_setup = f"""
<style>
  .block-container {{
    background-image: url("{pic_2}");
    background-size: cover;
    background-position: center;
    height: 98vh;
  }}
  div.block-container:nth-child(1)  div[data-testid="stHorizontalBlock"] {{
    height: 85vh;
  }}
  div.stButton > button:first-child {{
    background-color: #90CAF9;
    color:#ffffff;
  }}
  div.stButton > button:hover {{
    background-color: #E57373;
    color:#fff;
  }}
  div.stButton > button:focus:not(:active) {{
    background-color: #E57373;
    color: #fff;
    border:none;
  }}
</style>
"""

st.markdown(bg_setup, unsafe_allow_html=True)

##############################
####        LOGIC         ####
##############################

@st.cache_data
def load_data():
    data = pd.read_csv('tracks_clustered.csv')
    data['cluster'] = data['cluster'].astype(str)
    return data 

@st.cache_data
def load_top_100_data():
    data = pd.read_csv('billboard_100.csv')
    return data 

@st.cache_data
def apply_PCA(X, tracks, dim):
    pca = PCA(dim)
    sc = StandardScaler()
    X_prepped = sc.fit_transform(X)
    X_ = pca.fit_transform(X_prepped)
    
    df = pd.DataFrame(X_, columns=[*map(str, range(1, dim+1))])
    df['category_name'] = tracks['category_name']
    df['cluster'] = tracks['cluster']
    df['artists'] = tracks['artists']
    df['name'] = tracks['name']
    
    return df

def get_trackid_by_search_query(searchquery, dom_element):
    res = sp.search(searchquery)

    if len(res['tracks']['items']) > 0:
      id = res['tracks']['items'][0]['id'] 
      song = res['tracks']['items'][0]['name']
      artists = ', '.join([a['name'] for a in res['tracks']['items'][0]['artists']])
      return id, song, artists
    else:
      dom_element.warning('Nothing found for your query - please check your input')
      return '', '', ''

def get_audio_features(track_id):
   res = sp.audio_features([track_id])
   return res[0]
   
def get_top_recommendation(id):
   entry = data.query('track_id == @id')
   
   if entry.shape[0] == 1:
      cluster = entry['cluster'].iloc[0]
      filtered_data = data.query("(cluster == @cluster) & (track_id != @id)")
      fd = filtered_data.iloc[random.randint(1, filtered_data.shape[0] - 1)]
      song = fd['name']
      artists = fd['artists'] 
      return song, artists
   else:
      af = pd.json_normalize(get_audio_features(id))
      x = af[AUDIO_FEATURES]
      
      scaler = pd.read_pickle('fitted_scaler.pkl')
      model = pd.read_pickle('kmeans.pkl')

      pred = model.predict(scaler.transform(x))
      cluster = str(pred[0])
      filtered_data = data.query("cluster == @cluster")
      fd = filtered_data.iloc[random.randint(1, filtered_data.shape[0] - 1)]
      song = fd['name']
      artists = fd['artists'] 
      return song, artists

def get_top100_recommendation(search_query):
   art, sg = search_query.split('-')
   entry = top_100.query('(song_title.str.lower() == @sg.strip().lower()) and (artist.str.lower() == @art.strip().lower())')
   if entry.shape[0] == 1:
      rec = top_100.query('song_title.str.lower() != @sg.strip().lower()').iloc[random.randint(1, top_100.shape[0] - 1)]
      return rec['artist'], rec['song_title']
   else:
      return '', ''

def render_spotify_audio_features():
  with st.expander("More Information on Audio Features"):
    st.markdown('`acousticness` - A confidence measure from 0.0 to 1.0 of whether the track is acoustic. 1.0 represents high confidence the track is acoustic.')
    st.markdown('`danceability` - Danceability describes how suitable a track is for dancing based on a combination of musical elements including tempo, rhythm stability, beat strength, and overall regularity. A value of 0.0 is least danceable and 1.0 is most danceable.')

    st.markdown('`energy` - Energy is a measure from 0.0 to 1.0 and represents a perceptual measure of intensity and activity. Typically, energetic tracks feel fast, loud, and noisy. For example, death metal has high energy, while a Bach prelude scores low on the scale. Perceptual features contributing to this attribute include dynamic range, perceived loudness, timbre, onset rate, and general entropy.')
    st.markdown('`instrumentalness` - Predicts whether a track contains no vocals. "Ooh" and "aah" sounds are treated as instrumental in this context. Rap or spoken word tracks are clearly "vocal". The closer the instrumentalness value is to 1.0, the greater likelihood the track contains no vocal content. Values above 0.5 are intended to represent instrumental tracks, but confidence is higher as the value approaches 1.0.')
    st.markdown('`liveness` - Detects the presence of an audience in the recording. Higher liveness values represent an increased probability that the track was performed live. A value above 0.8 provides strong likelihood that the track is live.')
    st.markdown('`loudness` - The overall loudness of a track in decibels (dB). Loudness values are averaged across the entire track and are useful for comparing relative loudness of tracks. Loudness is the quality of a sound that is the primary psychological correlate of physical strength (amplitude). Values typically range between -60 and 0 db.')
    st.markdown('`speechiness` - Speechiness detects the presence of spoken words in a track. The more exclusively speech-like the recording (e.g. talk show, audio book, poetry), the closer to 1.0 the attribute value. Values above 0.66 describe tracks that are probably made entirely of spoken words. Values between 0.33 and 0.66 describe tracks that may contain both music and speech, either in sections or layered, including such cases as rap music. Values below 0.33 most likely represent music and other non-speech-like tracks.')
    st.markdown('`tempo` - The overall estimated tempo of a track in beats per minute (BPM). In musical terminology, tempo is the speed or pace of a given piece and derives directly from the average beat duration.')
    st.markdown('`valence` - A measure from 0.0 to 1.0 describing the musical positiveness conveyed by a track. Tracks with high valence sound more positive (e.g. happy, cheerful, euphoric), while tracks with low valence sound more negative (e.g. sad, depressed, angry).')


def handle_search_query(search_query, dom_element):
   if '-' not in search_query:
      dom_element.error('Please enter in right format: <artist> - <song>')
   elif len(search_query.split('-')) < 2 or len(search_query.split('-')[0]) < 1 or len(search_query.split('-')[1]) < 1:
      dom_element.error('Please enter in right format: <artist> - <song>') 
   else:
      artists, song = search_query.split('-')
      s, a = get_top100_recommendation(search_query)
      if not s:
          id, song, artists = get_trackid_by_search_query(f'q={artists} {song}&type=track', dom_element)
          if id:
             s, a = get_top_recommendation(id)

      dom_element.markdown(f'<div id="recommendation-box">🎵 Your Song:<br/> <b>{song}  --  {artists}</b><br/>\
                           <br/>  🎱 Recommendation: <br/><strong>{s}  --  {a}</strong></div>', unsafe_allow_html=True)
            
   
##############################
####        UI / HERO     ####
##############################
data = load_data()
top_100 = load_top_100_data()
data_PCA = apply_PCA(data[AUDIO_FEATURES], data, 3)

   
col1, col2, col3= st.columns([2, 3, 2])



col1.markdown('<span class="hide-mobile" style="color:#EF9A9A;font-size:28px;font-weight:700;">EXPLORING NEW SOUNDS</span>', unsafe_allow_html=True)
col1.title('MUS.IIC')
col1.text('Recommender')
col1.text('Explorer')
col1.text('Suggester')

col2.markdown('<div class="hide-mobile"><br/><br/><br/><br/><br/><br/><br/><br/><br/></div>', unsafe_allow_html=True)
search_query = col2.text_input('', placeholder='Enter a song you enjoy like: Linkin Park - Numb')
if col2.button('GET SONG!'):
   handle_search_query(search_query, col2)
   


##############################
####      UI / TABS       ####
##############################
tab1, tab2 = st.tabs(["Explore Dataset", "Song Clustering"])   

with tab1:
  ### section 0 ###
  st.write(' ')
  st.write(' ')
  st.subheader('Details on Dataset')
  st.markdown(f'Dataset was created by collecting song information from Spotify. It comprises {data.shape[0]:,.0f} tracks with \
              information on song name, artists, audio features, playlist with which the song was assosiated with during collection, \
              the category of the playlist and some more. Go to the **Raw Data** section below to have a look.') 
  
  ### section 1 ###
  st.write(' ')
  st.write(' ')
  st.subheader('Song categories')
  st.markdown(f'Visually explore songs by category - Based on PCA applied to only audio features of songs.')
  col_a, col_b = st.columns([1, 3])
  options_categories = col_a.multiselect('Select Categories', data['category_name'].unique(), ['Rock'],)
  fig = px.scatter_3d(data_PCA[data_PCA['category_name'].isin(options_categories)], x='1', y='2', z='3', color='category_name')
  fig.update_layout(height=800)
  col_b.plotly_chart(fig, use_container_width=True, height=800)

  ### section 2 ###
  st.subheader('Audio Features')
  st.write(' ')
  st.write(' ')
  st.markdown('Spotify generates the following audio features based on their audio anaylsis: `acousticness`  `danceability`  `energy`  \
              `instrumentalness`  `liveness`  `loudness`  `speechiness`  `tempo`  `valence`. These features are available for every song. \
              Click on the accordion below if you want to learn more about every one of them.')
  render_spotify_audio_features()
  st.write(' ')
  st.write(' ')
  st.write(' ')

  ### section 3 ###
  st.subheader('Spotify Songs - Raw Data')
  st.markdown('Sample of 100 entries from collected data used for clustering. Note that artists can have multiple entries seperated by comma.')
  st.write(data.loc[0:100, data.columns!='cluster'])

  

with tab2:
  ### section 0 ###
  st.write(' ')
  st.write(' ')
  st.subheader('Song Clustering Project')
  st.markdown(f'For the song recommender the collected songs from dataset were clustered with KMeans algorithm. For the clustering only audio features \
               were considered. See number of cluster development in the last section. <br/> Note that for the recommendation of songs in this app there \
               is a two step flow involved - **A)** it will check against billboard top 100 first and recommend another song from the list; if your entered song is **B)** not part of it, a song from the collected dataset will \
              be recommended which is assigned to the same cluster as the input', unsafe_allow_html=True)
  st.write(' ')
  st.write(' ')
  st.write(' ')
              
  ### section 1 ###
  st.subheader('Explore the Generated Clusters for Dataset')
  col_c, col_d = st.columns([1, 3])
  options = col_c.multiselect('Select Cluster', sorted(data['cluster'].unique()), ['0'],)
  fig_2 = px.scatter_3d(data_PCA[data_PCA['cluster'].isin(options)], x='1', y='2', z='3', color='cluster')
  fig_2.update_layout(height=800)
  col_d.plotly_chart(fig_2, use_container_width=True, height=800)

  ### section 2 ###
  st.subheader('Model Performance - Cluster Development')
  st.image('https://raw.githubusercontent.com/nataschaberg/song-recommender/master/k_clusters_kmeans_elbow.png', width=600)


##############################
####      CUSTOM CSS      ####
##############################

styl = f"""
<style>
  div[data-baseweb="tab-list"] p {{
    font-size: 20px;
    font-weight: 600;
    padding: 12px;
  }}
  div[data-baseweb="tab-list"] button:hover {{
    background-color: #fff;
  }}
  div[data-baseweb="tab-list"] button:focus {{
    background-color: #fff;
  }}
  div[data-baseweb="notification"] {{
    background-color: white;
    font-color: #000;
  }}
  div#recommendation-box {{
    background-color: #fff;
    padding: 12px;
    border-radius: 8px;
    box-shadow: rgba(100, 100, 111, 0.2) 0px 7px 29px 0px;
  }}
  @media (min-width: 450px) {{
    .hide-mobile {{
      display: block;
    }}
  }}
  @media (max-width: 450px) {{
    .hide-mobile {{
      display: none;
    }}
  }}
  footer {{visibility: hidden;}}
</style>
"""
st.markdown(styl, unsafe_allow_html=True)
