import streamlit as st
import pandas as pd
import plotly.express as px
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import pairwise_distances_argmin_min
import random

import spotipy
from spotipy.oauth2 import SpotifyClientCredentials

SPOTIPY_CLIENT_ID = st.secrets["secret_client_id"]
SPOTIPY_CLIENT_SECRET = st.secrets["spotipy_client_secret"]
AUDIO_FEATURES = ['danceability', 'energy', 'loudness', 'speechiness', 'acousticness', 'instrumentalness', 'liveness', 'valence', 'tempo']

auth_manager = SpotifyClientCredentials(client_id=SPOTIPY_CLIENT_ID, client_secret=SPOTIPY_CLIENT_SECRET)
sp = spotipy.Spotify(auth_manager=auth_manager)

st.set_page_config(layout="wide")
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

def get_trackid_by_search_query(searchquery):
    res = sp.search(searchquery)

    if len(res['tracks']['items']) > 0:
      id = res['tracks']['items'][0]['id'] 
      song = res['tracks']['items'][0]['name']
      artists = ', '.join([a['name'] for a in res['tracks']['items'][0]['artists']])
      return id, song, artists
    else:
       return '', '', ''

def get_audio_features(track_id):
   res = sp.audio_features([track_id])
   return res[0]
   
def get_top_recommendation(id):
   entry = data.query('track_id == @id')
   
   if entry.shape[0] == 1:
      cluster = entry['cluster'].iloc[0]
      filtered_data = data.query("(cluster == @cluster) & (track_id != @id)").sort_values(by=['popularity'], ascending=False).iloc[0]
      song = filtered_data['name']
      artists = filtered_data['artists'] 
      #closest, _ = pairwise_distances_argmin_min(entry[audio_features], data[data['cluster'] == entry['cluster']][audio_features])
      #col2.write(entry)
      return song, artists
   else:
      af = pd.json_normalize(get_audio_features(id))
      x = af[AUDIO_FEATURES]
      
      scaler = pd.read_pickle('fitted_scaler.pkl')
      model = pd.read_pickle('kmeans.pkl')

      pred = model.predict(scaler.transform(x))
      cluster = str(pred[0])
      filtered_data = data.query("cluster == @cluster").sort_values(by=['popularity'], ascending=False).iloc[0]
      song = filtered_data['name']
      artists = filtered_data['artists'] 
      return song, artists

data = load_data()
top_100 = load_top_100_data()
data_PCA = apply_PCA(data[AUDIO_FEATURES], data, 3)
pic_1 = 'https://images.pexels.com/photos/1037999/pexels-photo-1037999.jpeg?auto=compress&cs=tinysrgb&w=1260&h=750&dpr=2'
pic_2 = 'https://t3.ftcdn.net/jpg/04/14/62/04/360_F_414620426_iam5N6Bhp8YMOEOJD7MqdI1dC0Tw0CIY.jpg'
pic_3 = 'https://img.freepik.com/free-photo/headphones-mouse-orange-background_23-2148182104.jpg?w=2000&t=st=1678234576~exp=1678235176~hmac=0eefabf71f6411b037f71cc70153796c2e31c906efa4cba24b8235b74c1ea75b'

   
col1, col2, col3= st.columns([2, 3, 2])

col2.markdown('<br/><br/><br/><br/><br/><br/>', unsafe_allow_html=True)
col1.title('MUS.IIC')
col1.text('Recommender')
col1.text('Explorer')
col1.text('Suggester')

def get_top100_recommendation(search_query):
   art, sg = search_query.split('-')
   entry = top_100.query('(song_title.str.lower() == @sg.strip().lower()) and (artist.str.lower() == @art.strip().lower())')
   if entry.shape[0] == 1:
      rec = top_100.query('song_title.str.lower() != @sg.strip().lower()').iloc[random.randint(1, top_100.shape[0] - 1)]
      return rec['artist'], rec['song_title']
   else:
      return '', ''

search_query = col2.text_input('', 'artist - song')
if col2.button('Reccomend Song'):
   if '-' not in search_query:
      col2.error('Please enter in right format: <artist> - <song>')
   else:
      artists, song = search_query.split('-')
      s, a = get_top100_recommendation(search_query)
      if not s:
          id, song, artists = get_trackid_by_search_query(f'q={search_query}&type=track')
          if id:
             s, a = get_top_recommendation(id)
             col2.markdown(f'<div id="recommendation-box">🎵 Your Song:<br/> <b>{song}  --  {artists}</b><br/><br/>  🎱 Recommendation: <br/><strong>{s}  --  {a}</strong></div>', unsafe_allow_html=True)
          else:
             col2.warning('Nothing found for your entry - please check your input') 
      else:
         col2.markdown(f'<div id="recommendation-box">🎵 Your Song:<br/> <b>{song}  --  {artists}</b><br/><br/>  🎱 Recommendation: <br/><strong>{s}  --  {a}</strong></div>', unsafe_allow_html=True)
      



tab1, tab2 = st.tabs(["Explore Dataset", "Recommendation Insights"])

with tab1:
  st.subheader('Track Features based on Audio Analysis')
  st.write(' ')
  st.write(' ')
  st.markdown('Spotify conisders the following categories for audio analysis of tracks: `acousticness`  `danceability`  `energy`  `instrumentalness`  `liveness`  `loudness`  `speechiness`  `tempo`  `valence`')

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
   
   
  
  st.write(' ')
  st.write(' ')
  st.write(' ')
  st.subheader('Categories from inital playlists')

  col_a, col_b = st.columns([1, 3])
  options_categories = col_a.multiselect(
            'Initial Playlist Category',
            data['category_name'].unique(),
            ['Rock'],)

  fig = px.scatter_3d(data_PCA[data_PCA['category_name'].isin(options_categories)], x='1', y='2', z='3', color='category_name')
  fig.update_layout(height=800)
  col_b.plotly_chart(fig, use_container_width=True, height=800)

  st.markdown('----')
  st.subheader('Cluster after applying KMeans to audio features')
  col_c, col_d = st.columns([1, 3])
  options = col_c.multiselect(
            'Cluster after KMeans',
            sorted(data['cluster'].unique()),
            ['0'],)

  fig_2 = px.scatter_3d(data_PCA[data_PCA['cluster'].isin(options)], x='1', y='2', z='3', color='cluster')
  fig_2.update_layout(height=800)
  col_d.plotly_chart(fig_2, use_container_width=True, height=800)

  st.subheader('Spotify Tracks Collected Data')
  st.write(data.loc[0:100, :])

  

with tab2:
   st.title("A dog")
   st.image("https://static.streamlit.io/examples/dog.jpg", width=200)


styl = f"""
<style>
  .block-container {{
    background-image: url("{pic_2}");
    background-size: cover;
    height: 98vh;
  }}
  div[data-testid="stImage"] > img {{
    border-radius: 15px;
  }}
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
  div.stButton > button:first-child {{
    background-color: #1976D2;
    color:#ffffff;
  }}
  div.stButton > button:hover {{
    background-color: #90CAF9;
    color:#fff;
  }}
  div[data-baseweb="notification"] {{
    background-color: white;
    font-color: #000;
  }}
  div.block-container:nth-child(1)  div[data-testid="stHorizontalBlock"] {{
    height: 80vh;
  }}
  div#recommendation-box {{
    background-color: #fff;
    padding: 12px;
    border-radius: 8px;
  }}
  footer {{visibility: hidden;}}
</style>
"""
st.markdown(styl, unsafe_allow_html=True)
