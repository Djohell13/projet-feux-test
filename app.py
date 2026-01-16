#-------------------------------------------------------- Imports nécessaires ---------------------------------------------------
import os
import warnings
import pandas as pd
import numpy as np
import plotly.express as px
import matplotlib.pyplot as plt
import streamlit as st
import boto3
import mlflow
import mlflow.pyfunc
import folium
import base64
import streamlit.components.v1 as components
from PIL import Image
from io import StringIO
from branca.element import Template, MacroElement
from folium import DivIcon
from streamlit_folium import st_folium
from sklearn.exceptions import UndefinedMetricWarning
from sklearn import set_config
from sklearn.cluster import DBSCAN

warnings.filterwarnings("ignore", category=UndefinedMetricWarning)
set_config(display="text")

#--------------------------------------------------------- CSS -----------------------------------------------------------

if "css_injected" not in st.session_state:
    css_code = """
    <style>
        button[data-baseweb="tab"] {
            font-size: 20px !important;
            font-weight: 600;
            padding: 10px 18px;
        }
        
        button[data-baseweb="tab"][aria-selected="true"] {
            color: #1f77b4 !important;
            border-bottom: 3px solid #1f77b4 !important;
        }
        
        /* Réduit les micro-sauts et transitions inutiles */
        .stTabs [data-baseweb="tab-list"] {
            gap: 0px !important;
        }
        
        .element-container iframe,
        .stPlotlyChart {
            transition: none !important;
            animation: none !important;
        }
        
        /* Optionnel : masque le flash blanc initial */
        .stApp {
            background-color: white;  /* ou votre couleur de fond */
        }
    </style>
    """
    
    components.html(
        f"""
        <div style="display:none;">
            {css_code}
        </div>
        """,
        height=0
    )
    
    st.session_state.css_injected = True

#-------------------------------------------------------- Configuration AWS S3 ---------------------------------------------------
aws_access_key_id = st.secrets["AWS_ACCESS_KEY_ID"]
aws_secret_access_key = st.secrets["AWS_SECRET_ACCESS_KEY"]
aws_region = "eu-west-3"

s3 = boto3.client(
    's3',
    region_name=aws_region,
    aws_access_key_id=aws_access_key_id,
    aws_secret_access_key=aws_secret_access_key
)

#-------------------------------------------------------- Streamlit page config ---------------------------------------------------
st.set_page_config(page_title="Projet Incendies", page_icon="🔥", layout="wide")

# ------------------------- Charger l'image en base64 -------------------------

def get_image_base64(path):
    with open(path, "rb") as f:
        data = f.read()
    return base64.b64encode(data).decode()

logo_base64 = get_image_base64("images/logo.png")

# ------------------------- Header compact -------------------------
header = st.container()
with header:
    col_logo, col_title = st.columns([1, 3], vertical_alignment="center")
    with col_logo:
        st.image("images/logo.png", width=250)
    with col_title:
        st.markdown("""
            <h1 style="color:#FF4500; margin-bottom:0;"> 🔥 Projet Incendies </h1>
            <p style="font-size:16px; color:gray; margin-top:0;"> Analyse et prédiction des feux de forêts </p>
        """, unsafe_allow_html=True)
    st.markdown("---")

# ------------------------------- Tabs ---------------------------------------
content = st.container()
with content:

    page_tabs = st.tabs([
        "Notre Projet",
        "Exploration des données",
        "Notre démarche",
        "Notre application"
    ])

#-------------------------------------------------------- Footer ---------------------------------------------------
def show_footer():
    st.markdown("---")
    st.markdown("Projet réalisé dans le cadre de la formation Lead Data Scientist. © 2025")

#-------------------------------------------------------- Chargement des datasets (cachés) ---------------------------------------------------

def load_model_data() -> pd.DataFrame:
    bucket = "projet-final-lead"
    key = "data/dataset_complet_meteo.csv"
    try:
        obj = s3.get_object(Bucket=bucket, Key=key)
        df = pd.read_csv(obj['Body'], sep=';')
        for col in df.columns:
            if "date" in col.lower():
                df[col] = pd.to_datetime(df[col], errors="coerce", dayfirst=True)
        return df
    except Exception as e:
        st.error(f"❌ Erreur chargement dataset météo : {e}")
        return pd.DataFrame()

def load_casernes() -> pd.DataFrame:
    
    bucket = "projet-final-lead"
    key = "data/casernes_corses.csv"
    try:
        obj = s3.get_object(Bucket=bucket, Key=key)
        content = obj["Body"].read().decode("utf-8")
        df = pd.read_csv(StringIO(content), sep=",", encoding="utf-8")
        return df
    except Exception as e:
        st.error(f"❌ Erreur : {e}")
        return pd.DataFrame()

#-------------------------------------------------------- Chargement unique du modèle ---------------------------------------------------
mlflow.set_tracking_uri("https://djohell-ml-flow.hf.space")

if "model" not in st.session_state:
    with st.spinner("🔄 Chargement du modèle XGBoost Survival Cox"):
        model_uri = 'runs:/69a3c889954f4ce9a2139a4fb4cefc59/survival_xgb_model'
        st.session_state.model = mlflow.pyfunc.load_model(model_uri)

#-------------------------------------------------------- Prédiction optimisée par horizon ---------------------------------------------------

def predict_risk_for_horizon(df_raw: pd.DataFrame, horizon_days: int, _cache_key: str) -> pd.DataFrame:
    df = df_raw.copy()
    df = df.rename(columns={"Feu prévu": "event", "décompte": "duration"})
    df["event"] = df["event"].astype(bool)
    df["duration"] = df["duration"].fillna(0)

    features = [
        "moyenne precipitations mois", "moyenne temperature mois", "moyenne evapotranspiration mois",
        "moyenne vitesse vent année", "moyenne vitesse vent mois", "moyenne temperature année",
        "RR", "UM", "ETPMON", "TN", "TX", "Nombre de feu par an", "Nombre de feu par mois",
        "jours_sans_pluie", "jours_TX_sup_30", "ETPGRILLE_7j", "compteur jours vers prochain feu",
        "compteur feu log", "Année", "Mois", "moyenne precipitations année", "moyenne evapotranspiration année",
    ]
    features = [f for f in features if f in df.columns]

    log_hr = st.session_state.model.predict(df[features])
    HR = np.exp(log_hr)

    def S0(t): return np.exp(-t / 1000)

    col = f"proba_{horizon_days}j"
    df[col] = 1 - (S0(horizon_days) ** HR)

    for needed_col in ["latitude", "longitude", "ville"]:
        if needed_col not in df.columns:
            df[needed_col] = np.nan

    return df[["latitude", "longitude", "ville", col]].copy()

#-------------------------------------------------------- Tabs content ---------------------------------------------------

# Tab 0 - Notre Projet
with page_tabs[0]:
    st.title("🔥 Projet Analyse et prévention des Incendies 🔥")
    st.subheader("Bonjour et bienvenue dans notre projet de fin d'études mené chez Jedha ! " \
    "Nous sommes ravis de vous présenter notre travail sur l'analyse et la prédiction des feux de forêts en France, avec un focus particulier sur la Corse.")

    st.subheader(" 📊 Commençons par le contexte de ce projet et notamment l'importance des forêts françaises")
    st.subheader("🌲La forêt française en chiffres :")
    col1, col2 = st.columns([2,1])
    with col1:
        st.markdown("""
La France est le 4ᵉ pays européen en superficie forestière, avec **17,5 millions d’hectares** en métropole (32 % du territoire) et **8 millions** en Guyane.
Au total, les forêts couvrent environ **41 %** du territoire national.
- **75 %** des forêts sont privées (3,5 millions de propriétaires).
- **16 %** publiques (collectivités).
- **9 %** domaniales (État).
La forêt française est un réservoir de biodiversité :  
- **190 espèces d’arbres** (67 % feuillus, 33 % conifères).  
- **73 espèces de mammifères**, **120 d’oiseaux**.  
- Environ **30 000 espèces de champignons et autant d’insectes**.  
- **72 %** de la flore française se trouve en forêt.
Les forêts françaises absorbent environ **9 %** des émissions nationales de gaz à effet de serre, jouant un rôle crucial dans la lutte contre le changement climatique.
Le Code forestier encadre leur gestion durable pour protéger la biodiversité, l’air, l’eau et prévenir les risques naturels.
        """)

    st.header("🌲 Zoom sur le taux de boisement des différents départements français")

    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        st.image("images/boisement.png", caption="Taux de boisement par département en France en 2024 (source : IGN)", use_container_width=True)

    st.write("""---""")

    st.header(" Quels sont les principaux rôles de la forêt ?")
    col1, col2, col3 = st.columns([1,1,1])

    with col1:
        st.image("images/puits carbone.png", caption="Puits de carbone", width=400)

    with col2:
        st.image("images/racines.png", caption="Limitation de l'érosion", width=500)

    with col3:
        st.image("images/rayonnement.png", caption="Limitation du rayonnement solaire", width=400)

    st.write("""---""")

    st.markdown("""
    💡 **Synthèse** : La forêt joue un rôle essentiel pour le climat, le sol et la régulation du rayonnement solaire.  
    Elle absorbe le CO₂, limite l'érosion et protège la biodiversité.
    """)            
        
    st.subheader("👉 L'importance de la forêt n'est donc pas négligeable et les conséquences des feux sont lourdes :")

    st.markdown("""
    - 🔥 **Dégagement du carbone précédemment capturé**, aggravant le réchauffement climatique  
    - 🌱 **Destruction des écosystèmes** et perte de biodiversité  
    - 🌊 **Accentuation de l’érosion des sols** et risques d’inondations  
    - 🏠 **Impacts économiques et humains importants** (infrastructures, habitations, activités locales)""")

    st.markdown("""
    ### 👉 Nous avons ensuite décidé d'explorer les feux de forêts au niveau national afin de déterminer la zone principale qui fera l'objet de notre analyse prédictive
    """)

    st.write("""---""")

    st.subheader("🧭 Démarche et sources de données")

    st.markdown("""
    Notre projet repose sur une **analyse croisée de données environnementales et historiques**, avec pour objectif de mieux comprendre les facteurs déclencheurs des feux de forêts et d’en améliorer la prévision.

    ### 🔥 Données incendies — BDIFF
    Nous avons exploité les données issues de la **Base de Données sur les Incendies de Forêts en France (BDIFF)**, une source de référence nationale qui fournit :
    - la localisation des feux,
    - leur date de déclenchement,
    - leur surface brûlée,
    - ainsi que la **nature et l’origine des incendies** lorsqu’elles sont connues.

    ### 🌦️ Données météorologiques — Météo-France
    Afin d’intégrer l’influence des conditions climatiques, nous avons collecté des données météorologiques via l’**API Météo-France**, incluant notamment :
    - température,
    - précipitations,
    - humidité relative,
    - vitesse et direction du vent.

    ### 🔗 Enrichissement et modélisation
    Ces jeux de données ont ensuite été :
    - nettoyés,
    - harmonisés temporellement et spatialement,
    - puis fusionnés afin de construire un **dataset complet** servant de base à l’**analyse exploratoire**, au **clustering spatial** et à la **modélisation prédictive des risques de feux**.
    """)

#---------------------------------------------------- EDA ------------------------------------------------
with page_tabs[1]:
    st.subheader("🗺️ 1. Visualisation des incendies entre 2006 et 2024")

    image = Image.open("images/incendies.png")
    st.image(
    image,
    use_container_width=True
    )
    
    st.write("""---""")

    # ----------- DBSCAN Clustering -------------------------------
    st.subheader("🔥 2. Détection des clusters d'incendies")
    
    image = Image.open("images/clustering.png")
    st.image(
    image,
    use_container_width=True
    )

    st.info("""
    📌 **À retenir**  
    Ce graphique met en évidence une forte hétérogénéité territoriale.  
    Les départements les plus touchés concentrent une part disproportionnée des incendies, ce qui motive une approche prédictive ciblée.
    """)
    
    st.write("""---""")

    # ---------------------------------------------------------------
    # Histogramme mensuel
    st.subheader("3. Cumul des incendies mensuels par année")
    
    image = Image.open("images/saison.png")
    st.image(
    image,
    use_container_width=True
    )

    st.info(""" Sans grande surprise, on observe une saisonnalité marquée des incendies, avec un pic significatif durant les mois de juillet et août.
    """)

    st.write("""---""")

    st.subheader("4. Analyse des causes 🔥")

    image = Image.open("images/causes.png")
    st.image(
    image,
    use_container_width=True
    )

    st.info("""
    📌 **À retenir** : 
    L’analyse des causes révèle que la majorité des incendies sont d’origine humaine, soulignant l’importance de la prévention et de la sensibilisation pour réduire ces incidents.""")

    st.write("""---""")

        # ---------------------------------------------------- Nombre total d’incendies par année ----------------------------------
    image = Image.open("images/cumul_annuel.png")
    st.image(
    image,
    use_container_width=True
    )

    st.info("""L'année 2022 fut une année noire avec un nombre record d'incendies, notamment une surface brûlée de 60 000 ha, soit l'équivalent de 
                    84 000 terrains de football, ou encore 5,7 fois la surface de Paris intra-muros.""")

        # ---------------------------------------------------- Top départements
    st.subheader("6. Les 10 départements les plus touchés 🔥")

    image = Image.open("images/departement.png")
    st.image(
    image,
    caption="Top 10 des départements les plus touchés par les incendies",
    use_container_width=True
    )

    st.info("""Après analyse, la Corse se démarque nettement comme le département le plus touché par les incendies de forêt en France, justifiant ainsi notre choix de focus pour la suite de notre projet.""")

    show_footer()

#-------------------------------------------------------- Page Notre démarche ---------------------------------------------------
with page_tabs[2]:
    st.header("🔥 Corse : Bilan Campagne Feux de Forêts 2024")
    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs(["📌 Contexte","🛠️ Prévention","🚒 Moyens","📊 Statistiques","🔍 Causes","🔎 Enquêtes"])

        # Contenus des tabs
    with tab1:
            with st.expander("📌 Contexte général"):
                st.markdown("""
    - **80 %** de la Corse est couverte de forêts/maquis → **fort risque incendie**  
    - **2023-2024** : la plus chaude et la plus sèche jamais enregistrée  
    - **714 mm** de pluie sur l’année (**78 %** de la normale)  
    - **Façade orientale** : seulement **30 %** des précipitations normales
                """)

    with tab2:
            with st.expander("🛠️ Prévention & Investissements"):
                st.markdown("""
    - **1,9 million €** investis en 2023-2024 par l’État (jusqu’à 80 % de financement)  
    - Travaux financés :  
    - Pistes DFCI/DECI (Sorio di Tenda, Oletta, Île-Rousse…)  
    - Citernes souples & points d’eau  
    - Drones, caméras thermiques, logiciels SIG  
    - Véhicules pour réserves communales
                """)

    with tab3:
            with st.expander("🚒 Moyens déployés"):
                st.markdown("""
    - Jusqu’à **500 personnels mobilisables**  
    - **168 sapeurs-pompiers SIS2B**, **261 UIISC5**, forestiers-sapeurs, gendarmerie, ONF…  
    - Moyens aériens :  
    - **1 hélico**, **2 canadairs** à Ajaccio  
    - **12 canadairs** + **8 Dashs** nationaux en renfort
                """)

    with tab4:
            with st.expander("📊 Statistiques Feux Été 2024"):
                st.markdown("""
    - **107 feux** recensés (~9/semaine)  
    - **130 ha** brûlés dont :  
    - 83 % des feux <1 ha : **5,42 ha**  
    - 4 gros feux >10 ha : **72,84 ha**  
    - Linguizetta (**22,19 ha**), Oletta (**18,9 ha**), Pioggiola (**18,75 ha**), Tallone (**13 ha**)  
    - Depuis janvier 2024 : **285 feux** pour **587 ha**  
    - Feu majeur à Barbaggio : **195 ha (33 % du total annuel)**
                """)

    with tab5:
            with st.expander("🔍 Causes des feux (38 cas identifiés)"):
                st.markdown("""
    - **11** : foudre  
    - **8** : écobuages  
    - **6** : malveillance  
    - **5** : accidents  
    - **4** : mégots de cigarette
                """)
            with st.expander("⚠️ Prévention = priorité absolue"):
                st.markdown("""
    - **90 %** des feux ont une origine humaine  
    - Causes principales : **imprudences** (mégots, BBQ, travaux, écobuages…)
                """)

    with tab6:
            with st.expander("🔎 Enquêtes & Surveillance"):
                st.markdown("""
    - **20 incendies** étudiés par la Cellule Technique d’Investigation (CTIFF)  
    - Équipes mobilisées : **7 forestiers**, **15 pompiers**, **21 forces de l’ordre**  
    - **Fermeture de massif** enclenchée 1 seule fois : forêt de Pinia
                """)

    #---------------------------------------------------Notre Objectif --------------------------------------------------------
    
    st.subheader("🎯 Notre Objectif")
    st.markdown("""
    Dans un contexte de **changement climatique** et de **risques accrus d’incendies de forêt**, notre équipe a développé un projet innovant visant à **analyser et prédire les zones à risque d’incendie** en France, avec un focus particulier sur la **Corse**.
        """)
    #---------------------------------------------------Obectifs du projet---------------------------------------------------
    col1, col2 = st.columns([1, 1])
    with col1:
            st.subheader("🔍 Exploration des données")
            st.markdown("""
    - ✅ **Évolution du nombre d’incendies**, répartition par mois et par causes.
    - ✅ **Cartographie interactive** des incendies sur tout le territoire.
    - ✅ **Analyse des clusters** grâce à DBSCAN pour identifier les zones les plus à risque.
            """)

    with col2:
            st.subheader("📈 Modèles prédictifs")
            st.markdown("""
    - ✅ **Comparaison des modèles** : Random Forest, XGBoost, analyse de survie.
    - ✅ **Prédiction des zones à risque** avec visualisation sur carte.
    - ✅ Fourniture d'un **outil décisionnel** pour les autorités et les services de gestion des risques.
            """)

    st.subheader("📘 Définition de l'analyse de survie (Survival Analysis")
    col1, col2, col3, col4 = st.columns(4)

    with col1:
            st.markdown("### 🧠 Qu’est-ce que l’analyse de survie ?")
            st.markdown("""
    L’**analyse de survie** (ou **Survival Analysis**) est une méthode statistique utilisée pour **modéliser le temps avant qu’un événement se produise**, comme :
    - 🔥 un incendie,
    - 🏥 un décès,
    - 📉 une résiliation d’abonnement,
    - 🧯 une panne.
    """)

    with col2:
            st.markdown("### 📌 Objectif :")
            st.markdown("""
    > Estimer la **probabilité qu’un événement ne se soit pas encore produit** à un instant donné.
    """)

    with col3:
            st.markdown("### 🔑 Concepts fondamentaux : ")
            st.markdown("""
    - ⏳ **Temps de survie (`T`)** : temps écoulé jusqu’à l’événement.
    - 🎯 **Événement** : le phénomène qu’on cherche à prédire (feu, panne, décès...).
    - ❓ **Censure** : l’événement **n’a pas encore eu lieu** durant la période d’observation.
    - 📉 **Fonction de survie `S(t)`** : probabilité de "survivre" après le temps `t`.
    - ⚠️ **Fonction de risque `h(t)`** : probabilité que l’événement se produise **immédiatement après `t`**, sachant qu’il ne s’est pas encore produit.
    """)
        
    with col4:
            st.markdown ("### 🧪 Exemples d’applications :")
            st.markdown("""
    | Domaine | Exemple |
    |--------|---------|
    | 🔥 Incendies | Quand un feu va-t-il se déclarer ? |
    | 🏥 Santé | Combien de temps un patient survivra après traitement ? |
    | 📉 Marketing | Quand un client risque-t-il de partir ? |
    | 🧑‍💼 RH | Quand un salarié quittera-t-il l’entreprise ? |
    """)

    #---------------------------------------------------Equipe du projet---------------------------------------------------
    st.subheader("👨‍💻 Équipe du projet")
    col1, col2, col3, col4, col5, col6 = st.columns(6)
    with col1:
            st.image("images/Francois_Minaret.jpg", width=150)
            st.markdown("**Francois Minaret**")
    with col2:
            st.image("images/Joel_Termondjian.jpg", width=150)  
            st.markdown("**Joël Termondjian**")
    with col3:
            st.image("images/Marc_Barthes.jpg", width=150)
            st.markdown("**Marc Barthes**")
    with col4:
            st.image("images/Gilles_Akakpo.jpg", width=150)
            st.markdown("**Gilles Akakpo**")
    with col5:
            st.image("images/Nathalie_Devogelaere.jpg", width=150)
            st.markdown("**Nathalie Devogelaere**")
    with col6:
            st.image("images/David_Jaoui.jpg", width=150)
            st.markdown("**David Jaoui**")

        
    show_footer()


#-------------------------------------------------------- Page Notre application ---------------------------------------------------
with page_tabs[3]:
    st.header("Carte du risque d’incendie en Corse")

    df_raw = load_model_data()

    horizons_lbl = {
        "7 jours": 7,
        "30 jours": 30,
        "60 jours": 60,
        "90 jours": 90,
        "180 jours": 180,
    }

    choix = st.radio(
        "Choisissez l’horizon temporel souhaité :",
        list(horizons_lbl.keys()),
        horizontal=True,
        index=0,
        key="horizon_risk_select"   # ← clé importante !
    )

    horizon_days = horizons_lbl[choix]
    cache_key = f"risk_pred_{horizon_days}"

    df_map = predict_risk_for_horizon(df_raw, horizon_days, cache_key)

    col_proba = f"proba_{horizon_days}j"
    vmax = float(df_map[col_proba].max() or 1.0)

    fig = px.scatter_mapbox(
        df_map,
        lat="latitude",
        lon="longitude",
        hover_name="ville",
        hover_data={col_proba: ":.2%"},
        color=col_proba,
        color_continuous_scale="YlOrRd",
        range_color=(0.0, vmax),
        zoom=7,
        height=650,
    )
    fig.update_layout(
        mapbox_style="open-street-map",
        margin=dict(l=0, r=0, t=0, b=0),
        coloraxis_colorbar=dict(title="Probabilité", tickformat=".0%"),
    )

    st.subheader(f"Risque d’incendie – horizon **{choix}**")
    st.plotly_chart(fig, use_container_width=True, key=f"risk_map_{horizon_days}")

    # Carte des casernes
    st.subheader("🗺️ Carte des casernes et équipements de lutte contre les incendies")
    df_casernes = load_casernes()

    if not df_casernes.empty:
        df_casernes['latitude'] = df_casernes['latitude'].astype(str).str.replace(',', '.').astype(float)
        df_casernes['longitude'] = df_casernes['longitude'].astype(str).str.replace(',', '.').astype(float)
        df_casernes = df_casernes.dropna(subset=['latitude', 'longitude'])

        df_casernes['categorie'] = np.select(
            [
                df_casernes['nom'].str.contains('centre', case=False, na=False),
                df_casernes['nom'].str.contains('base', case=False, na=False),
                df_casernes['nom'].str.contains('SSLIA', case=False, na=False),
                df_casernes['nom'].str.contains('citerne', case=False, na=False),
                df_casernes['nom'].str.contains('borne', case=False, na=False),
            ],
            ["Centre d'incendie et de secours", 'Base forestière', 'SSLIA (aérodromes)', 'Citerne', 'Borne incendie'],
            default='Autre'
        )

        emoji_legende = {
            "Centre d'incendie et de secours": "🚒",
            "Base forestière": "🌲",
            "SSLIA (aérodromes)": "✈️",
            "Citerne": "💦"
        }

        m = folium.Map(location=[42.0396, 9.0129], zoom_start=8)

        for _, row in df_casernes.iterrows():
            emoji = emoji_legende.get(row['categorie'], '❓')
            folium.Marker(
                location=[row['latitude'], row['longitude']],
                popup=f"{emoji} {row['nom']}",
                icon=DivIcon(html=f"""<div style="font-size:24px">{emoji}</div>""")
            ).add_to(m)

        # Légende
        legend_html = """
        {% macro html(this, kwargs) %}
            <div style="
                position: fixed; bottom: 50px; left: 50px; width: 280px;
                background-color: white; border: 2px solid grey; z-index: 9999;
                font-size: 14px; color: black; padding: 10px; border-radius: 10px;
                box-shadow: 2px 2px 6px rgba(0,0,0,0.3);
            ">
                <b>📘 Légende</b><br>
                🚒 Centre d'incendie et de secours<br>
                🌲 Base forestière<br>
                ✈️ SSLIA (aérodromes)<br>
                💦 Citerne<br>
            </div>
        {% endmacro %}
        """
        legend = MacroElement()
        legend._template = Template(legend_html)
        m.get_root().add_child(legend)

        st_folium(m, width=1200, height=650, key="casernes_map")

    # Comparaison modèles
    st.subheader("📈 Comparaison des modèles prédictifs de survie")
    st.markdown("""
    | Modèle                               | Concordance Index |
    |--------------------------------------|-------------------|
    | Predict survival fonction (Baseline) | 0.69              |
    | XGBOOST survival cox                 | 0.809             |
    """)
    st.markdown("👉 Le modèle **XGBOOST survival cox** obtient la meilleure performance globale.")

    show_footer()
