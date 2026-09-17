"""
Innovative Feature 3: Multi-Language Localization Engine (i18n).
Provides dictionary lookup for English, Hindi, Spanish, and French dashboard UI elements.
"""

TRANSLATIONS = {
    "English": {
        "title": "Natural Disaster Intelligence & Response Dashboard",
        "subtitle": "Real-time Monitoring, Predictive Modeling & Emergency Response Operations",
        "tab_overview": "📊 Executive Overview",
        "tab_map": "🗺️ Interactive Hazard Map",
        "tab_analytics": "📈 Historical Trends & Analytics",
        "tab_ml": "🤖 ML Hazard Predictions",
        "tab_citizen": "👤 Citizen Portal & Safety",
        "tab_government": "🏛️ Government Operations",
        "tab_innovations": "🚀 Innovative Tools",
        "kpi_total_events": "Total Disaster Events",
        "kpi_critical_alerts": "Critical Warnings",
        "kpi_casualties": "Total Casualties",
        "kpi_affected": "Affected Population",
        "kpi_loss": "Economic Loss ($M)",
        "filter_region": "Select Region",
        "filter_disaster": "Disaster Type",
        "btn_refresh": "🔄 Refresh Real-Time Data"
    },
    "Hindi": {
        "title": "प्राकृतिक आपदा बुद्धिमत्ता एवं प्रतिक्रिया डैशबोर्ड",
        "subtitle": "वास्तविक समय निगरानी, पूर्वानुमान मॉडल और आपातकालीन प्रतिक्रिया",
        "tab_overview": "📊 कार्यकारी अवलोकन",
        "tab_map": "🗺️ इंटरएक्टिव जोखिम मानचित्र",
        "tab_analytics": "📈 ऐतिहासिक रुझान",
        "tab_ml": "🤖 एमएल आपदा पूर्वानुमान",
        "tab_citizen": "👤 नागरिक पोर्टल एवं सुरक्षा",
        "tab_government": "🏛️ सरकारी संसाधन संचालन",
        "tab_innovations": "🚀 नवीन उपकरण",
        "kpi_total_events": "कुल आपदा घटनाएं",
        "kpi_critical_alerts": "गंभीर चेतावनियां",
        "kpi_casualties": "कुल हताहत",
        "kpi_affected": "प्रभावित जनसंख्या",
        "kpi_loss": "आर्थिक नुकसान ($ मिलियन)",
        "filter_region": "क्षेत्र चुनें",
        "filter_disaster": "आपदा का प्रकार",
        "btn_refresh": "🔄 डाटा रिफ्रेश करें"
    },
    "Spanish": {
        "title": "Panel de Inteligencia y Respuesta ante Desastres Naturales",
        "subtitle": "Monitoreo en Tiempo Real, Modelado Predictivo y Operaciones de Emergencia",
        "tab_overview": "📊 Resumen Ejecutivo",
        "tab_map": "🗺️ Mapa Interactivo de Riesgos",
        "tab_analytics": "📈 Tendencias Históricas",
        "tab_ml": "🤖 Predicciones de IA/ML",
        "tab_citizen": "👤 Portal del Ciudadano",
        "tab_government": "🏛️ Gestión Gubernamental",
        "tab_innovations": "🚀 Herramientas Innovadoras",
        "kpi_total_events": "Eventos Totales",
        "kpi_critical_alerts": "Alertas Críticas",
        "kpi_casualties": "Víctimas Totales",
        "kpi_affected": "Población Afectada",
        "kpi_loss": "Pérdida Económica ($M)",
        "filter_region": "Seleccionar Región",
        "filter_disaster": "Tipo de Desastre",
        "btn_refresh": "🔄 Actualizar Datos"
    },
    "French": {
        "title": "Tableau de Bord d'Intelligence sur les Catastrophes Naturelles",
        "subtitle": "Surveillance en Temps Réel, Modélisation Prédictive et Intervention d'Urgence",
        "tab_overview": "📊 Aperçu Exécutif",
        "tab_map": "🗺️ Carte Interactive des Risques",
        "tab_analytics": "📈 Tendances Historiques",
        "tab_ml": "🤖 Prédictions par IA",
        "tab_citizen": "👤 Portail Citoyen & Sécurité",
        "tab_government": "🏛️ Opérations Gouvernementales",
        "tab_innovations": "🚀 Outils Innovants",
        "kpi_total_events": "Total des Événements",
        "kpi_critical_alerts": "Alertes Critiques",
        "kpi_casualties": "Nombre de Victimes",
        "kpi_affected": "Population Touchée",
        "kpi_loss": "Pertes Économiques ($M)",
        "filter_region": "Sélectionner la Région",
        "filter_disaster": "Type de Catastrophe",
        "btn_refresh": "🔄 Rafraîchir les Données"
    }
}


def get_text(key: str, lang: str = "English") -> str:
    """Retrieve localized string for given UI element key."""
    lang_dict = TRANSLATIONS.get(lang, TRANSLATIONS["English"])
    return lang_dict.get(key, TRANSLATIONS["English"].get(key, key))
