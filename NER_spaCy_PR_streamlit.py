#!/usr/bin/env python3
"""
Streamlit Entity Linker Application with Pelagios Integration

A web interface for Named Entity Recognition using spaCy with enhanced
linking to Pelagios network services including Peripleo, Pleiades, and 
Recogito-compatible exports.

Author: EntityLinker Team
Version: 2.0 - Enhanced with Pelagios Integration
"""

import streamlit as st

# Configure Streamlit page FIRST - before any other Streamlit commands
st.set_page_config(
    page_title="From Text to Linked Data using spaCy + Pelagios",
    layout="centered",  
    initial_sidebar_state="collapsed" 
)

# Authentication is REQUIRED - do not run app without proper login
try:
    import streamlit_authenticator as stauth
    import yaml
    from yaml.loader import SafeLoader
    import os
    
    # Check if config file exists
    if not os.path.exists('config.yaml'):
        st.error("Authentication required: config.yaml file not found!")
        st.info("Please ensure config.yaml is in the same directory as this app.")
        st.stop()
    
    # Load configuration
    with open('config.yaml') as file:
        config = yaml.load(file, Loader=SafeLoader)

    # Setup authentication
    authenticator = stauth.Authenticate(
        config['credentials'],
        config['cookie']['name'],
        config['cookie']['key'],
        config['cookie']['expiry_days']
    )

    # Check if already authenticated via session state
    if 'authentication_status' in st.session_state and st.session_state['authentication_status']:
        name = st.session_state['name']
        authenticator.logout("Logout", "sidebar")
        # Continue to app below...
    else:
        # Render login form
        try:
            # Try different login methods
            login_result = None
            try:
                login_result = authenticator.login(location='main')
            except TypeError:
                try:
                    login_result = authenticator.login('Login', 'main')
                except TypeError:
                    login_result = authenticator.login()
            
            # Handle the result
            if login_result is None:
                # Check session state for authentication result
                if 'authentication_status' in st.session_state:
                    auth_status = st.session_state['authentication_status']
                    if auth_status == False:
                        st.error("Username/password is incorrect")
                        st.info("Try username: demo_user with your password")
                    elif auth_status == None:
                        st.warning("Please enter your username and password")
                    elif auth_status == True:
                        st.rerun()  # Refresh to show authenticated state
                else:
                    st.warning("Please enter your username and password")
                st.stop()
            elif isinstance(login_result, tuple) and len(login_result) == 3:
                name, auth_status, username = login_result
                # Store in session state
                st.session_state['authentication_status'] = auth_status
                st.session_state['name'] = name
                st.session_state['username'] = username
                
                if auth_status == True:
                    st.rerun()  # Refresh to show authenticated state
                elif auth_status == False:
                    st.error("Username/password is incorrect")
                    st.stop()
            else:
                st.error(f"Unexpected login result format: {login_result}")
                st.stop()
                
        except Exception as login_error:
            st.error(f"Login method error: {login_error}")
            st.stop()
        
except ImportError:
    st.error("Authentication required: streamlit-authenticator not installed!")
    st.info("Please install streamlit-authenticator to access this application.")
    st.stop()
except Exception as e:
    st.error(f"Authentication error: {e}")
    st.info("Cannot proceed without proper authentication.")
    st.stop()

import streamlit.components.v1 as components
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import json
import io
import base64
from typing import List, Dict, Any, Optional
import sys
import os
import requests
import time
import xml.etree.ElementTree as ET
from datetime import datetime
import re
import urllib.parse
import hashlib


class PelagiosIntegration:
    """Integration class for Pelagios services."""
    
    def __init__(self):
        """Initialize Pelagios integration."""
        self.peripleo_base_url = "https://peripleo.pelagios.org/api"
        self.pleiades_base_url = "https://pleiades.stoa.org"
        
    def search_peripleo(self, place_name: str, limit: int = 5) -> List[Dict]:
        """
        Search Peripleo for historical place information.
        
        Args:
            place_name: Name of place to search
            limit: Maximum number of results
            
        Returns:
            List of place records from Peripleo
        """
        try:
            url = f"{self.peripleo_base_url}/search"
            params = {
                'q': place_name,
                'limit': limit,
                'type': 'place'
            }
            headers = {'User-Agent': 'EntityLinker-Pelagios/2.0'}
            
            response = requests.get(url, params=params, headers=headers, timeout=10)
            if response.status_code == 200:
                data = response.json()
                return data.get('items', [])
            
        except Exception as e:
            print(f"Peripleo search failed for {place_name}: {e}")
            return []
        
        return []
    
    def enhance_entities_with_pelagios(self, entities: List[Dict]) -> List[Dict]:
        """
        Enhance place entities with Pelagios data.
        
        Args:
            entities: List of extracted entities
            
        Returns:
            Enhanced entities with Pelagios links
        """
        place_types = ['GPE', 'LOCATION', 'FACILITY']
        
        for entity in entities:
            if entity['type'] in place_types:
                # Search Peripleo for this place
                peripleo_results = self.search_peripleo(entity['text'])
                
                if peripleo_results:
                    best_match = peripleo_results[0]  # Take first result
                    
                    # Add Pelagios data to entity
                    entity['pelagios_data'] = {
                        'peripleo_id': best_match.get('identifier'),
                        'peripleo_title': best_match.get('title'),
                        'peripleo_description': best_match.get('description'),
                        'temporal_bounds': best_match.get('temporal_bounds'),
                        'place_types': best_match.get('place_types', []),
                        'peripleo_url': f"https://peripleo.pelagios.org/ui#selected={best_match.get('identifier')}"
                    }
                    
                    # Extract coordinates if available
                    if best_match.get('geo_bounds'):
                        bounds = best_match['geo_bounds']
                        if 'centroid' in bounds:
                            entity['latitude'] = bounds['centroid']['lat']
                            entity['longitude'] = bounds['centroid']['lon']
                            entity['geocoding_source'] = 'pelagios_peripleo'
                    
                    # Add Pleiades ID if available
                    if best_match.get('source_gazetteer') == 'pleiades':
                        pleiades_id = best_match.get('source_id')
                        if pleiades_id:
                            entity['pleiades_id'] = pleiades_id
                            entity['pleiades_url'] = f"{self.pleiades_base_url}/places/{pleiades_id}"
                
                time.sleep(0.2)  # Rate limiting
        
        return entities
    
    def export_to_recogito_format(self, text: str, entities: List[Dict], 
                                 title: str = "EntityLinker Export") -> str:
        """
        Export entities in Recogito-compatible JSON format.
        
        Args:
            text: Original text
            entities: List of entities
            title: Document title
            
        Returns:
            JSON string in Recogito format
        """
        # Recogito annotation format
        annotations = []
        
        for idx, entity in enumerate(entities):
            # Only export place entities for Recogito
            if entity['type'] in ['GPE', 'LOCATION', 'FACILITY', 'ADDRESS']:
                annotation = {
                    "@id": f"annotation_{idx}",
                    "@type": "Annotation",
                    "body": [
                        {
                            "type": "TextualBody",
                            "purpose": "tagging",
                            "value": entity['type']
                        }
                    ],
                    "target": {
                        "source": title,
                        "selector": {
                            "type": "TextQuoteSelector",
                            "exact": entity['text'],
                            "start": entity['start'],
                            "end": entity['end']
                        }
                    },
                    "created": datetime.now().isoformat()
                }
                
                # Add place identification if available
                if entity.get('pleiades_id'):
                    annotation["body"].append({
                        "type": "SpecificResource",
                        "purpose": "identifying",
                        "source": {
                            "id": entity['pleiades_url'],
                            "label": entity['text']
                        }
                    })
                elif entity.get('wikidata_url'):
                    annotation["body"].append({
                        "type": "SpecificResource", 
                        "purpose": "identifying",
                        "source": {
                            "id": entity['wikidata_url'],
                            "label": entity['text']
                        }
                    })
                
                # Add coordinates if available
                if entity.get('latitude') and entity.get('longitude'):
                    annotation["body"].append({
                        "type": "GeometryBody",
                        "purpose": "geotagging",
                        "geometry": {
                            "type": "Point",
                            "coordinates": [entity['longitude'], entity['latitude']]
                        }
                    })
                
                annotations.append(annotation)
        
        # Create Recogito document format
        recogito_export = {
            "@context": "http://www.w3.org/ns/anno.jsonld",
            "id": f"recogito_export_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            "type": "AnnotationCollection",
            "label": title,
            "created": datetime.now().isoformat(),
            "generator": {
                "id": "https://github.com/your-repo/entity-linker",
                "type": "Software",
                "name": "EntityLinker with spaCy + Pelagios"
            },
            "total": len(annotations),
            "first": {
                "id": f"{title}_page1",
                "type": "AnnotationPage", 
                "items": annotations
            }
        }
        
        return json.dumps(recogito_export, indent=2, ensure_ascii=False)
    
    def export_to_tei_xml(self, text: str, entities: List[Dict], 
                         title: str = "EntityLinker Export") -> str:
        """
        Export annotated text as TEI XML with place markup.
        
        Args:
            text: Original text
            entities: List of entities
            title: Document title
            
        Returns:
            TEI XML string
        """
        # Create TEI structure
        tei_ns = "http://www.tei-c.org/ns/1.0"
        ET.register_namespace('', tei_ns)
        
        root = ET.Element("{%s}TEI" % tei_ns)
        root.set("xmlns", tei_ns)
        
        # TEI Header
        header = ET.SubElement(root, "{%s}teiHeader" % tei_ns)
        file_desc = ET.SubElement(header, "{%s}fileDesc" % tei_ns)
        title_stmt = ET.SubElement(file_desc, "{%s}titleStmt" % tei_ns)
        title_elem = ET.SubElement(title_stmt, "{%s}title" % tei_ns)
        title_elem.text = title
        
        pub_stmt = ET.SubElement(file_desc, "{%s}publicationStmt" % tei_ns)
        pub_stmt_p = ET.SubElement(pub_stmt, "{%s}p" % tei_ns)
        pub_stmt_p.text = "Generated by EntityLinker with Pelagios Integration"
        
        # Text body
        text_elem = ET.SubElement(root, "{%s}text" % tei_ns)
        body = ET.SubElement(text_elem, "{%s}body" % tei_ns)
        div = ET.SubElement(body, "{%s}div" % tei_ns)
        
        # Process text with entity markup
        annotated_text = self._create_tei_markup(text, entities, tei_ns)
        div.append(annotated_text)
        
        # Convert to string
        xml_str = ET.tostring(root, encoding='unicode', method='xml')
        
        # Add XML declaration and pretty format
        pretty_xml = '<?xml version="1.0" encoding="UTF-8"?>\n' + xml_str
        
        return pretty_xml
    
    def _create_tei_markup(self, text: str, entities: List[Dict], tei_ns: str):
        """Create TEI markup with place tags."""
        # Sort entities by start position
        sorted_entities = sorted(entities, key=lambda x: x['start'])
        
        # Create paragraph element
        p = ET.Element("{%s}p" % tei_ns)
        
        current_pos = 0
        
        for entity in sorted_entities:
            # Add text before entity
            if entity['start'] > current_pos:
                if len(p) == 0 and p.text is None:
                    p.text = text[current_pos:entity['start']]
                else:
                    # Add to tail of last element
                    if len(p) > 0:
                        if p[-1].tail is None:
                            p[-1].tail = text[current_pos:entity['start']]
                        else:
                            p[-1].tail += text[current_pos:entity['start']]
            
            # Create place name element based on entity type
            if entity['type'] in ['GPE', 'LOCATION']:
                place_elem = ET.SubElement(p, "{%s}placeName" % tei_ns)
            elif entity['type'] == 'PERSON':
                place_elem = ET.SubElement(p, "{%s}persName" % tei_ns)
            elif entity['type'] == 'ORGANIZATION':
                place_elem = ET.SubElement(p, "{%s}orgName" % tei_ns)
            else:
                place_elem = ET.SubElement(p, "{%s}name" % tei_ns)
            
            place_elem.text = entity['text']
            
            # Add attributes
            if entity.get('pleiades_id'):
                place_elem.set('ref', entity['pleiades_url'])
            elif entity.get('wikidata_url'):
                place_elem.set('ref', entity['wikidata_url'])
            
            if entity.get('latitude') and entity.get('longitude'):
                place_elem.set('geo', f"{entity['latitude']},{entity['longitude']}")
            
            current_pos = entity['end']
        
        # Add remaining text
        if current_pos < len(text):
            if len(p) == 0:
                p.text = (p.text or "") + text[current_pos:]
            else:
                p[-1].tail = (p[-1].tail or "") + text[current_pos:]
        
        return p
    
    def create_pelagios_map_url(self, entities: List[Dict]) -> str:
        """
        Create a Peripleo map URL showing all georeferenced entities.
        
        Args:
            entities: List of entities with coordinates
            
        Returns:
            URL to Peripleo map view
        """
        # Filter entities with coordinates
        geo_entities = [e for e in entities if e.get('latitude') and e.get('longitude')]
        
        if not geo_entities:
            return "https://peripleo.pelagios.org"
        
        # Calculate bounding box
        lats = [e['latitude'] for e in geo_entities]
        lons = [e['longitude'] for e in geo_entities]
        
        min_lat, max_lat = min(lats), max(lats)
        min_lon, max_lon = min(lons), max(lons)
        
        # Add some padding
        lat_padding = (max_lat - min_lat) * 0.1 or 0.01
        lon_padding = (max_lon - min_lon) * 0.1 or 0.01
        
        bbox = f"{min_lon-lon_padding},{min_lat-lat_padding},{max_lon+lon_padding},{max_lat+lat_padding}"
        
        return f"https://peripleo.pelagios.org/ui#bbox={bbox}"


class EntityLinker:
    """
    Main class for entity linking functionality with Pelagios integration.
    
    This class handles the complete pipeline from text processing to entity
    extraction, validation, linking, and output generation.
    """
    
    def __init__(self):
        """Initialize the EntityLinker and load required spaCy model."""
        self.nlp = self._load_spacy_model()
        
        # Color scheme for different entity types in HTML output
        self.colors = {
            'PERSON': '#BF7B69',          # F&B Red earth        
            'ORGANIZATION': '#9fd2cd',    # F&B Blue ground
            'GPE': '#C4C3A2',             # F&B Cooking apple green
            'LOCATION': '#EFCA89',        # F&B Yellow ground. 
            'FACILITY': '#C3B5AC',        # F&B Elephants breath
            'GSP': '#C4A998',             # F&B Dead salmon
            'ADDRESS': '#CCBEAA'          # F&B Oxford stone
        }
    
    def _load_spacy_model(self):
        """Load spaCy model with error handling and automatic download."""
        import spacy
        
        # Try to load models in order of preference
        models_to_try = ['en_core_web_sm', 'en_core_web_md', 'en_core_web_lg']
        
        for model_name in models_to_try:
            try:
                nlp = spacy.load(model_name)
                return nlp
            except OSError:
                continue
        
        # If no model is available, try to download en_core_web_sm
        st.info("No spaCy model found. Attempting to download en_core_web_sm...")
        try:
            import subprocess
            import sys
            
            # Download the model
            subprocess.check_call([sys.executable, "-m", "spacy", "download", "en_core_web_sm"])
            
            # Try to load it
            nlp = spacy.load("en_core_web_sm")
            st.success("Successfully downloaded and loaded en_core_web_sm")
            return nlp
            
        except Exception as e:
            st.error(f"Failed to download spaCy model: {e}")
            st.error("Please install a spaCy English model manually:")
            st.code("python -m spacy download en_core_web_sm")
            st.stop()

    def extract_entities(self, text: str):
        """Extract named entities from text using spaCy with proper validation."""
        # Process text with spaCy
        doc = self.nlp(text)
        
        entities = []
        
        # Step 1: Extract traditional named entities with validation
        for ent in doc.ents:
            # Filter out unwanted entity types at the spaCy label level
            if ent.label_ in ['DATE', 'TIME', 'PERCENT', 'MONEY', 'QUANTITY', 'ORDINAL', 'CARDINAL']:
                continue
            
            # Map spaCy entity types to our format
            entity_type = self._map_spacy_entity_type(ent.label_)
            
            # Additional filter in case mapping returns an unwanted type
            if entity_type in ['DATE', 'TIME', 'PERCENT', 'MONEY', 'QUANTITY', 'ORDINAL', 'CARDINAL']:
                continue
            
            # Validate entity using grammatical context
            if self._is_valid_entity(ent.text, entity_type, ent):
                entities.append({
                    'text': ent.text,
                    'type': entity_type,
                    'start': ent.start_char,
                    'end': ent.end_char,
                    'label': ent.label_  # Keep original spaCy label for reference
                })
        
        # Step 2: Extract addresses
        addresses = self._extract_addresses(text)
        entities.extend(addresses)
        
        # Step 3: Remove overlapping entities
        entities = self._remove_overlapping_entities(entities)
        
        return entities

    def _map_spacy_entity_type(self, spacy_label: str) -> str:
        """Map spaCy entity labels to our standardized types."""
        mapping = {
            'PERSON': 'PERSON',
            'ORG': 'ORGANIZATION',
            'GPE': 'GPE',  # Geopolitical entity
            'LOC': 'LOCATION',
            'FAC': 'FACILITY',
            'NORP': 'GPE',  # Nationalities or religious or political groups -> GPE
            'EVENT': 'LOCATION',  # Events often have location relevance
            'WORK_OF_ART': 'ORGANIZATION',  # Often associated with organizations
            'LAW': 'ORGANIZATION',  # Laws often associated with organizations
            'LANGUAGE': 'GPE'  # Languages associated with places
        }
        return mapping.get(spacy_label, spacy_label)

    def _is_valid_entity(self, entity_text: str, entity_type: str, spacy_ent) -> bool:
        """Validate an entity using spaCy's linguistic features."""
        # Skip very short entities
        if len(entity_text.strip()) <= 1:
            return False
        
        # Use spaCy's built-in confidence and linguistic features
        doc = spacy_ent.doc
        
        # Get the token(s) for this entity
        entity_tokens = [token for token in doc[spacy_ent.start:spacy_ent.end]]
        
        if not entity_tokens:
            return True  # Default to valid if we can't analyze
        
        first_token = entity_tokens[0]
        
        # Filter out words functioning as verbs or adjectives primarily
        if first_token.pos_ in ['VERB', 'AUX'] or (first_token.pos_ == 'ADJ' and entity_type == 'PERSON'):
            return False
        
        # Prefer proper nouns as they're more likely to be real entities
        if first_token.pos_ == 'PROPN':
            return True
        
        return True

    def _extract_addresses(self, text: str):
        """Extract address patterns that NER might miss."""
        import re
        addresses = []
        
        # Patterns for different address formats
        address_patterns = [
            r'\b\d{1,4}[-–]\d{1,4}\s+[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\s+(?:Road|Street|Avenue|Lane|Drive|Way|Place|Square|Gardens)\b',
            r'\b\d{1,4}\s+[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\s+(?:Road|Street|Avenue|Lane|Drive|Way|Place|Square|Gardens)\b'
        ]
        
        for pattern in address_patterns:
            for match in re.finditer(pattern, text):
                addresses.append({
                    'text': match.group(),
                    'type': 'ADDRESS',
                    'start': match.start(),
                    'end': match.end()
                })
        
        return addresses

    def _remove_overlapping_entities(self, entities):
        """Remove overlapping entities, keeping the longest ones."""
        entities.sort(key=lambda x: x['start'])
        
        filtered = []
        for entity in entities:
            overlaps = False
            for existing in filtered[:]:  # Create a copy to safely modify during iteration
                # Check if entities overlap
                if (entity['start'] < existing['end'] and entity['end'] > existing['start']):
                    # If current entity is longer, remove the existing one
                    if len(entity['text']) > len(existing['text']):
                        filtered.remove(existing)
                        break
                    else:
                        # Current entity is shorter, skip it
                        overlaps = True
                        break
            
            if not overlaps:
                filtered.append(entity)
        
        return filtered

    def get_coordinates(self, entities):
        """Enhanced coordinate lookup with geographical context detection."""
        import requests
        import time
        
        # Detect geographical context from the full text
        context_clues = self._detect_geographical_context(
            st.session_state.get('processed_text', ''), 
            entities
        )
        
        place_types = ['GPE', 'LOCATION', 'FACILITY', 'ORGANIZATION', 'ADDRESS']
        
        for entity in entities:
            if entity['type'] in place_types:
                # Skip if already has coordinates
                if entity.get('latitude') is not None:
                    continue
                
                # Try geocoding with context
                if self._try_contextual_geocoding(entity, context_clues):
                    continue
                    
                # Fall back to original methods
                if self._try_python_geocoding(entity):
                    continue
                
                if self._try_openstreetmap(entity):
                    continue
                    
                # If still no coordinates, try a more aggressive search
                self._try_aggressive_geocoding(entity)
        
        return entities
    
    def _detect_geographical_context(self, text: str, entities: List[Dict[str, Any]]) -> List[str]:
        """Detect geographical context from the text to improve geocoding accuracy."""
        import re
        
        context_clues = []
        text_lower = text.lower()
        
        # Extract major cities/countries mentioned in the text
        major_locations = {
            # Countries
            'uk': ['uk', 'united kingdom', 'britain', 'great britain'],
            'usa': ['usa', 'united states', 'america', 'us '],
            'canada': ['canada'],
            'australia': ['australia'],
            'france': ['france'],
            'germany': ['germany'],
            'italy': ['italy'],
            'spain': ['spain'],
            'japan': ['japan'],
            'china': ['china'],
            'india': ['india'],
            
            # Major cities that provide strong context
            'london': ['london'],
            'new york': ['new york', 'nyc', 'manhattan'],
            'paris': ['paris'],
            'tokyo': ['tokyo'],
            'sydney': ['sydney'],
            'toronto': ['toronto'],
            'berlin': ['berlin'],
            'rome': ['rome'],
            'madrid': ['madrid'],
            'beijing': ['beijing'],
            'mumbai': ['mumbai'],
            'los angeles': ['los angeles', 'la ', ' la,'],
            'chicago': ['chicago'],
            'boston': ['boston'],
            'edinburgh': ['edinburgh'],
            'glasgow': ['glasgow'],
            'manchester': ['manchester'],
            'birmingham': ['birmingham'],
            'liverpool': ['liverpool'],
            'bristol': ['bristol'],
            'leeds': ['leeds'],
            'cardiff': ['cardiff'],
            'belfast': ['belfast'],
            'dublin': ['dublin'],
        }
        
        # Check for explicit mentions
        for location, patterns in major_locations.items():
            for pattern in patterns:
                if pattern in text_lower:
                    context_clues.append(location)
                    break
        
        return context_clues[:3]  # Return top 3 context clues

    def _try_contextual_geocoding(self, entity, context_clues):
        """Try geocoding with geographical context."""
        import requests
        import time
        
        if not context_clues:
            return False
        
        # Create context-aware search terms
        search_variations = [entity['text']]
        
        # Add context to search terms
        for context in context_clues:
            context_mapping = {
                'uk': ['UK', 'United Kingdom', 'England', 'Britain'],
                'usa': ['USA', 'United States', 'US'],
                'canada': ['Canada'],
                'australia': ['Australia'],
                'france': ['France'],
                'germany': ['Germany'],
                'london': ['London, UK', 'London, England'],
                'new york': ['New York, USA', 'New York, NY'],
                'paris': ['Paris, France'],
                'tokyo': ['Tokyo, Japan'],
                'sydney': ['Sydney, Australia'],
            }
            
            context_variants = context_mapping.get(context, [context])
            for variant in context_variants:
                search_variations.append(f"{entity['text']}, {variant}")
        
        # Remove duplicates while preserving order
        search_variations = list(dict.fromkeys(search_variations))
        
        # Try geopy first with context
        try:
            from geopy.geocoders import Nominatim
            from geopy.exc import GeocoderTimedOut, GeocoderServiceError
            
            geocoder = Nominatim(user_agent="EntityLinker/2.0", timeout=10)
            
            for search_term in search_variations[:5]:  # Try top 5 variations
                try:
                    location = geocoder.geocode(search_term, timeout=10)
                    if location:
                        entity['latitude'] = location.latitude
                        entity['longitude'] = location.longitude
                        entity['location_name'] = location.address
                        entity['geocoding_source'] = f'geopy_contextual'
                        entity['search_term_used'] = search_term
                        return True
                    
                    time.sleep(0.2)  # Rate limiting
                except (GeocoderTimedOut, GeocoderServiceError):
                    continue
                    
        except ImportError:
            pass
        
        return False
    
    def _try_python_geocoding(self, entity):
        """Try Python geocoding libraries (geopy) - original method."""
        try:
            from geopy.geocoders import Nominatim, ArcGIS
            from geopy.exc import GeocoderTimedOut, GeocoderServiceError
            
            geocoders = [
                ('nominatim', Nominatim(user_agent="EntityLinker/2.0", timeout=10)),
                ('arcgis', ArcGIS(timeout=10)),
            ]
            
            for name, geocoder in geocoders:
                try:
                    location = geocoder.geocode(entity['text'], timeout=10)
                    if location:
                        entity['latitude'] = location.latitude
                        entity['longitude'] = location.longitude
                        entity['location_name'] = location.address
                        entity['geocoding_source'] = f'geopy_{name}'
                        return True
                        
                    time.sleep(0.3)
                except (GeocoderTimedOut, GeocoderServiceError):
                    continue
                except Exception as e:
                    continue
                        
        except ImportError:
            pass
        except Exception as e:
            pass
        
        return False
    
    def _try_openstreetmap(self, entity):
        """Fall back to direct OpenStreetMap Nominatim API."""
        try:
            url = "https://nominatim.openstreetmap.org/search"
            params = {
                'q': entity['text'],
                'format': 'json',
                'limit': 1,
                'addressdetails': 1
            }
            headers = {'User-Agent': 'EntityLinker/2.0'}
        
            response = requests.get(url, params=params, headers=headers, timeout=10)
            if response.status_code == 200:
                data = response.json()
                if data:
                    result = data[0]
                    entity['latitude'] = float(result['lat'])
                    entity['longitude'] = float(result['lon'])
                    entity['location_name'] = result['display_name']
                    entity['geocoding_source'] = 'openstreetmap'
                    return True
        
            time.sleep(0.3)  # Rate limiting
        except Exception as e:
            print(f"OpenStreetMap geocoding failed for {entity['text']}: {e}")
            pass
        
        return False
    
    def _try_aggressive_geocoding(self, entity):
        """Try more aggressive geocoding with different search terms."""
        import requests
        import time
        
        # Try variations of the entity name
        search_variations = [
            entity['text'],
            f"{entity['text']}, UK",  # Add country for UK places
            f"{entity['text']}, England",
            f"{entity['text']}, Scotland",
            f"{entity['text']}, Wales",
            f"{entity['text']} city",
            f"{entity['text']} town"
        ]
        
        for search_term in search_variations:
            try:
                url = "https://nominatim.openstreetmap.org/search"
                params = {
                    'q': search_term,
                    'format': 'json',
                    'limit': 1,
                    'addressdetails': 1
                }
                headers = {'User-Agent': 'EntityLinker/2.0'}
            
                response = requests.get(url, params=params, headers=headers, timeout=10)
                if response.status_code == 200:
                    data = response.json()
                    if data:
                        result = data[0]
                        entity['latitude'] = float(result['lat'])
                        entity['longitude'] = float(result['lon'])
                        entity['location_name'] = result['display_name']
                        entity['geocoding_source'] = f'openstreetmap_aggressive'
                        entity['search_term_used'] = search_term
                        return True
            
                time.sleep(0.2)  # Rate limiting between attempts
            except Exception:
                continue
        
        return False

    def link_to_wikidata(self, entities):
        """Add basic Wikidata linking."""
        import requests
        import time
        
        for entity in entities:
            try:
                url = "https://www.wikidata.org/w/api.php"
                params = {
                    'action': 'wbsearchentities',
                    'format': 'json',
                    'search': entity['text'],
                    'language': 'en',
                    'limit': 1,
                    'type': 'item'
                }
                
                response = requests.get(url, params=params, timeout=5)
                if response.status_code == 200:
                    data = response.json()
                    if data.get('search') and len(data['search']) > 0:
                        result = data['search'][0]
                        entity['wikidata_url'] = f"http://www.wikidata.org/entity/{result['id']}"
                        entity['wikidata_description'] = result.get('description', '')
                
                time.sleep(0.1)  # Rate limiting
            except Exception:
                pass  # Continue if API call fails
        
        return entities

    def link_to_wikipedia(self, entities):
        """Add Wikipedia linking for entities without Wikidata links."""
        import requests
        import time
        import urllib.parse
        
        for entity in entities:
            # Skip if already has Wikidata link
            if entity.get('wikidata_url'):
                continue
                
            try:
                # Use Wikipedia's search API
                search_url = "https://en.wikipedia.org/w/api.php"
                search_params = {
                    'action': 'query',
                    'format': 'json',
                    'list': 'search',
                    'srsearch': entity['text'],
                    'srlimit': 1
                }
                
                headers = {'User-Agent': 'EntityLinker/2.0'}
                response = requests.get(search_url, params=search_params, headers=headers, timeout=10)
                
                if response.status_code == 200:
                    data = response.json()
                    if data.get('query', {}).get('search'):
                        # Get the first search result
                        result = data['query']['search'][0]
                        page_title = result['title']
                        
                        # Create Wikipedia URL
                        encoded_title = urllib.parse.quote(page_title.replace(' ', '_'))
                        entity['wikipedia_url'] = f"https://en.wikipedia.org/wiki/{encoded_title}"
                        entity['wikipedia_title'] = page_title
                        
                        # Get a snippet/description from the search result
                        if result.get('snippet'):
                            # Clean up the snippet (remove HTML tags)
                            import re
                            snippet = re.sub(r'<[^>]+>', '', result['snippet'])
                            entity['wikipedia_description'] = snippet[:200] + "..." if len(snippet) > 200 else snippet
                
                time.sleep(0.2)  # Rate limiting
            except Exception as e:
                print(f"Wikipedia linking failed for {entity['text']}: {e}")
                pass
        
        return entities

    def link_to_britannica(self, entities):
        """Add basic Britannica linking.""" 
        import requests
        import re
        import time
        
        for entity in entities:
            # Skip if already has Wikidata or Wikipedia link
            if entity.get('wikidata_url') or entity.get('wikipedia_url'):
                continue
                
            try:
                search_url = "https://www.britannica.com/search"
                params = {'query': entity['text']}
                headers = {
                    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
                }
                
                response = requests.get(search_url, params=params, headers=headers, timeout=10)
                if response.status_code == 200:
                    # Look for article links
                    pattern = r'href="(/topic/[^"]*)"[^>]*>([^<]*)</a>'
                    matches = re.findall(pattern, response.text)
                    
                    for url_path, link_text in matches:
                        if (entity['text'].lower() in link_text.lower() or 
                            link_text.lower() in entity['text'].lower()):
                            entity['britannica_url'] = f"https://www.britannica.com{url_path}"
                            entity['britannica_title'] = link_text.strip()
                            break
                
                time.sleep(0.3)  # Rate limiting
            except Exception:
                pass
        
        return entities


class StreamlitEntityLinker:
    """
    Streamlit wrapper for the EntityLinker class with Pelagios integration.
    
    Provides a web interface with additional visualization and
    export capabilities for entity analysis.
    """
    
    def __init__(self):
        """Initialize the Streamlit Entity Linker."""
        self.entity_linker = EntityLinker()
        self.pelagios = PelagiosIntegration()
        
        # Initialize session state
        if 'entities' not in st.session_state:
            st.session_state.entities = []
        if 'processed_text' not in st.session_state:
            st.session_state.processed_text = ""
        if 'html_content' not in st.session_state:
            st.session_state.html_content = ""
        if 'analysis_title' not in st.session_state:
            st.session_state.analysis_title = "text_analysis"
        if 'last_processed_hash' not in st.session_state:
            st.session_state.last_processed_hash = ""

    @st.cache_data
    def cached_extract_entities(_self, text: str) -> List[Dict[str, Any]]:
        """Cached entity extraction to avoid reprocessing same text."""
        return _self.entity_linker.extract_entities(text)
    
    @st.cache_data  
    def cached_link_to_wikidata(_self, entities_json: str) -> str:
        """Cached Wikidata linking."""
        import json
        entities = json.loads(entities_json)
        linked_entities = _self.entity_linker.link_to_wikidata(entities)
        return json.dumps(linked_entities, default=str)
    
    @st.cache_data
    def cached_link_to_britannica(_self, entities_json: str) -> str:
        """Cached Britannica linking."""
        import json
        entities = json.loads(entities_json)
        linked_entities = _self.entity_linker.link_to_britannica(entities)
        return json.dumps(linked_entities, default=str)

    @st.cache_data
    def cached_link_to_wikipedia(_self, entities_json: str) -> str:
        """Cached Wikipedia linking."""
        import json
        entities = json.loads(entities_json)
        linked_entities = _self.entity_linker.link_to_wikipedia(entities)
        return json.dumps(linked_entities, default=str)

    @st.cache_data
    def cached_enhance_with_pelagios(_self, entities_json: str) -> str:
        """Cached Pelagios enhancement."""
        import json
        entities = json.loads(entities_json)
        enhanced_entities = _self.pelagios.enhance_entities_with_pelagios(entities)
        return json.dumps(enhanced_entities, default=str)

    def render_header(self):
        """Render the application header with logo."""
        # Display logo if it exists
        try:
            # Try to load and display the logo
            logo_path = "logo.png"  # You can change this filename as needed
            if os.path.exists(logo_path):
                # Logo naturally aligns to the left without columns
                st.image(logo_path, width=300)  # Adjust width as needed
            else:
                # If logo file doesn't exist, show a placeholder or message
                st.info("Place your logo.png file in the same directory as this app to display it here")
        except Exception as e:
            # If there's any error loading the logo, continue without it
            st.warning(f"Could not load logo: {e}")        
        # Add some spacing after logo
        st.markdown("<br>", unsafe_allow_html=True)
        
        # Main title and description
        st.header("From Text to Linked Data using spaCy + Pelagios")
        st.markdown("**Extract and link named entities from text to external knowledge bases including historical gazetteers**")
        
        # Create a simple process diagram
        st.markdown("""
        <div style="background-color: white; padding: 20px; border-radius: 10px; margin: 20px 0; border: 1px solid #E0D7C0;">
            <div style="text-align: center; margin-bottom: 20px;">
                <div style="background-color: #C4C3A2; padding: 10px; border-radius: 5px; display: inline-block; margin: 5px;">
                     <strong>Input Text</strong>
                </div>
                <div style="margin: 10px 0;">⬇️</div>
                <div style="background-color: #9fd2cd; padding: 10px; border-radius: 5px; display: inline-block; margin: 5px;">
                     <strong>spaCy Entity Recognition</strong>
                </div>
                <div style="margin: 10px 0;">⬇️</div>
                <div style="text-align: center;">
                    <strong>Link to Knowledge Bases:</strong>
                </div>
                <div style="margin: 15px 0;">
                    <div style="background-color: #EFCA89; padding: 8px; border-radius: 5px; display: inline-block; margin: 3px; font-size: 0.9em;">
                         <strong>Pelagios/Peripleo</strong><br><small>Historical geography</small>
                    </div>
                    <div style="background-color: #C3B5AC; padding: 8px; border-radius: 5px; display: inline-block; margin: 3px; font-size: 0.9em;">
                        <strong>Pleiades</strong><br><small>Ancient world gazetteer</small>
                    </div>
                    <div style="background-color: #BF7B69; padding: 8px; border-radius: 5px; display: inline-block; margin: 3px; font-size: 0.9em;">
                         <strong>Wikidata/Wikipedia</strong><br><small>General knowledge</small>
                    </div>
                </div>
                <div style="margin: 10px 0;">⬇️</div>
                <div style="text-align: center;">
                    <strong>Output Formats:</strong>
                </div>
                <div style="margin: 15px 0;">
                    <div style="background-color: #E8E1D4; padding: 8px; border-radius: 5px; display: inline-block; margin: 3px; font-size: 0.9em; border: 2px solid #EFCA89;">
                         <strong>Recogito JSON</strong><br><small>Collaborative annotation</small>
                    </div>
                    <div style="background-color: #E8E1D4; padding: 8px; border-radius: 5px; display: inline-block; margin: 3px; font-size: 0.9em; border: 2px solid #C3B5AC;">
                         <strong>TEI XML</strong><br><small>Digital humanities</small>
                    </div>
                    <div style="background-color: #E8E1D4; padding: 8px; border-radius: 5px; display: inline-block; margin: 3px; font-size: 0.9em; border: 2px solid #BF7B69;">
                         <strong>JSON-LD</strong><br><small>Linked data format</small>
                    </div>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)

    def render_sidebar(self):
        """Render the sidebar with information."""
        # Entity linking information
        st.sidebar.subheader("Entity Linking")
        st.sidebar.info("Entities are linked to Pelagios (historical places), Wikidata, Wikipedia, and Britannica as fallbacks. Places are geocoded using multiple services.")
        
        # Pelagios information
        st.sidebar.subheader("Pelagios Integration")
        st.sidebar.info("Historical places are enhanced with data from Peripleo and linked to authoritative gazetteers like Pleiades for ancient world geography.")
        
        # Export formats
        st.sidebar.subheader("Export Formats")
        st.sidebar.info("Results can be exported as Recogito annotations for collaborative markup, TEI XML for digital humanities, or JSON-LD for linked data applications.")

    def render_input_section(self):
        """Render the text input section."""
        st.header("Input Text")
        
        # Add title input
        analysis_title = st.text_input(
            "Analysis Title (optional)",
            placeholder="Enter a title for this analysis...",
            help="This will be used for naming output files"
        )
        
        # Sample text for demonstration - Marco Polo sample
        sample_text = """When we departed from Constantinople, we journeyed through Armenia Minor and came to the great city of Trebizond on the shores of the Black Sea. From thence we traveled inland through Erzurum and across the high passes of the Armenian mountains until we reached the ancient city of Tabriz in Persia, where merchants gather from all corners of the known world.

From Tabriz we proceeded to Baghdad, that great city on the river Tigris where the Caliph once held sway over all the lands of Islam. Though much diminished since the coming of the Mongol host, Baghdad still serves as a crossroads for caravans traveling between Damascus and the cities of India. We observed there the ruins of great palaces and learned from local merchants of the trade routes that lead to Samarkand and Bukhara, those jewels of the Silk Road.

Our path then led us through the desert of Khorasan to the city of Balkh, which the ancients called the Mother of Cities. Here Alexander the Great established his northern capital, and here too the armies of Genghis Khan laid waste to temples and bazaars. From Balkh we crossed the Oxus River and climbed into the Pamir Mountains, that roof of the world where no trees grow and the air is so thin that fire burns pale and weak."""       
        
        # Text input area - always shown and editable
        text_input = st.text_area(
            "Enter your text here:",
            value=sample_text,  # Pre-populate with sample text
            height=200,  # Reduced height for mobile
            placeholder="Paste your text here for entity extraction...",
            help="You can edit this text or replace it with your own content. This sample shows Marco Polo's travels - perfect for testing Pelagios integration!"
        )
        
        # File upload option in expander for mobile
        with st.expander("Or upload a text file"):
            uploaded_file = st.file_uploader(
                "Choose a text file",
                type=['txt', 'md'],
                help="Upload a plain text file (.txt) or Markdown file (.md) to replace the text above"
            )
            
            if uploaded_file is not None:
                try:
                    uploaded_text = str(uploaded_file.read(), "utf-8")
                    text_input = uploaded_text  # Override the text area content
                    st.success(f"File uploaded successfully! ({len(uploaded_text)} characters)")
                    # Set default title from filename if no title provided
                    if not analysis_title:
                        import os
                        default_title = os.path.splitext(uploaded_file.name)[0]
                        st.session_state.suggested_title = default_title
                except Exception as e:
                    st.error(f"Error reading file: {e}")
        
        # Use suggested title if no title provided
        if not analysis_title and hasattr(st.session_state, 'suggested_title'):
            analysis_title = st.session_state.suggested_title
        elif not analysis_title and not uploaded_file:
            analysis_title = "marco_polo_travels"
        
        return text_input, analysis_title or "text_analysis"

    def process_text(self, text: str, title: str):
        """
        Process the input text using the EntityLinker with Pelagios integration.
        
        Args:
            text: Input text to process
            title: Analysis title
        """
        if not text.strip():
            st.warning("Please enter some text to analyze.")
            return
        
        # Check if we've already processed this exact text
        text_hash = hashlib.md5(text.encode()).hexdigest()
        
        if text_hash == st.session_state.last_processed_hash:
            st.info("This text has already been processed. Results shown below.")
            return
        
        with st.spinner("Processing text and extracting entities..."):
            try:
                # Create a progress bar for the different steps
                progress_bar = st.progress(0)
                status_text = st.empty()
                
                # Step 1: Extract entities (cached)
                status_text.text("Extracting entities with spaCy...")
                progress_bar.progress(15)
                entities = self.cached_extract_entities(text)
                
                # Step 2: Link to Wikidata (cached)
                status_text.text("Linking to Wikidata...")
                progress_bar.progress(30)
                entities_json = json.dumps(entities, default=str)
                linked_entities_json = self.cached_link_to_wikidata(entities_json)
                entities = json.loads(linked_entities_json)
                
                # Step 3: Link to Wikipedia (cached)
                status_text.text("Linking to Wikipedia...")
                progress_bar.progress(45)
                entities_json = json.dumps(entities, default=str)
                linked_entities_json = self.cached_link_to_wikipedia(entities_json)
                entities = json.loads(linked_entities_json)
                
                # Step 4: Link to Britannica (cached)
                status_text.text("Linking to Britannica...")
                progress_bar.progress(55)
                entities_json = json.dumps(entities, default=str)
                linked_entities_json = self.cached_link_to_britannica(entities_json)
                entities = json.loads(linked_entities_json)
                
                # Step 5: Enhance with Pelagios (NEW!)
                status_text.text("Enhancing with Pelagios historical data...")
                progress_bar.progress(70)
                entities_json = json.dumps(entities, default=str)
                enhanced_entities_json = self.cached_enhance_with_pelagios(entities_json)
                entities = json.loads(enhanced_entities_json)
                
                # Step 6: Get coordinates
                status_text.text("Getting coordinates...")
                progress_bar.progress(85)
                entities = self.entity_linker.get_coordinates(entities)
                
                # Step 7: Generate visualization
                status_text.text("Generating visualization...")
                progress_bar.progress(100)
                html_content = self.create_highlighted_html(text, entities)
                
                # Store in session state
                st.session_state.entities = entities
                st.session_state.processed_text = text
                st.session_state.html_content = html_content
                st.session_state.analysis_title = title
                st.session_state.last_processed_hash = text_hash
                
                # Clear progress indicators
                progress_bar.empty()
                status_text.empty()
                
                # Show summary
                pelagios_enhanced = len([e for e in entities if e.get('pelagios_data')])
                geocoded = len([e for e in entities if e.get('latitude')])
                
                st.success(f"Processing complete! Found {len(entities)} entities")
                if pelagios_enhanced > 0:
                    st.info(f"{pelagios_enhanced} places enhanced with Pelagios data")
                if geocoded > 0:
                    st.info(f"{geocoded} places geocoded with coordinates")
                
            except Exception as e:
                st.error(f"Error processing text: {e}")
                st.exception(e)

    def create_highlighted_html(self, text: str, entities: List[Dict[str, Any]]) -> str:
        """
        Create HTML content with highlighted entities for display.
        
        Args:
            text: Original text
            entities: List of entity dictionaries
            
        Returns:
            HTML string with highlighted entities
        """
        import html as html_module
        
        # Sort entities by start position (reverse for safe replacement)
        sorted_entities = sorted(entities, key=lambda x: x['start'], reverse=True)
        
        # Start with escaped text
        highlighted = html_module.escape(text)
        
        # Color scheme (updated for spaCy entity types)
        colors = {
            'PERSON': '#BF7B69',          # F&B Red earth        
            'ORGANIZATION': '#9fd2cd',    # F&B Blue ground
            'GPE': '#C4C3A2',             # F&B Cooking apple green
            'LOCATION': '#EFCA89',        # F&B Yellow ground 
            'FACILITY': '#C3B5AC',        # F&B Elephants breath
            'GSP': '#C4A998',             # F&B Dead salmon
            'ADDRESS': '#CCBEAA'          # F&B Oxford stone
        }
        
        # Replace entities from end to start
        for entity in sorted_entities:
            # Highlight entities that have links OR coordinates OR Pelagios data
            has_links = (entity.get('britannica_url') or 
                         entity.get('wikidata_url') or 
                         entity.get('wikipedia_url') or     
                         entity.get('openstreetmap_url') or
                         entity.get('pleiades_url'))
            has_coordinates = entity.get('latitude') is not None
            has_pelagios = entity.get('pelagios_data') is not None
            
            if not (has_links or has_coordinates or has_pelagios):
                continue
                
            start = entity['start']
            end = entity['end']
            original_entity_text = text[start:end]
            escaped_entity_text = html_module.escape(original_entity_text)
            color = colors.get(entity['type'], '#E7E2D2')
            
            # Create tooltip with entity information
            tooltip_parts = [f"Type: {entity['type']}"]
            if entity.get('pelagios_data'):
                pelagios_data = entity['pelagios_data']
                if pelagios_data.get('peripleo_title'):
                    tooltip_parts.append(f"Peripleo: {pelagios_data['peripleo_title']}")
                if pelagios_data.get('temporal_bounds'):
                    tooltip_parts.append(f"Period: {pelagios_data['temporal_bounds']}")
            if entity.get('wikidata_description'):
                tooltip_parts.append(f"Description: {entity['wikidata_description']}")
            if entity.get('location_name'):
                tooltip_parts.append(f"Location: {entity['location_name']}")
            
            tooltip = " | ".join(tooltip_parts)
            
            # Create highlighted span with link (priority: Pleiades > Peripleo > Wikipedia > Wikidata > Britannica > OpenStreetMap)
            if entity.get('pleiades_url'):
                url = html_module.escape(entity["pleiades_url"])
                replacement = f'<a href="{url}" style="background-color: {color}; padding: 2px 4px; border-radius: 3px; text-decoration: none; color: black; border: 2px solid #8B4513;" target="_blank" title="{tooltip}">{escaped_entity_text}</a>'
            elif entity.get('pelagios_data', {}).get('peripleo_url'):
                url = html_module.escape(entity["pelagios_data"]["peripleo_url"])
                replacement = f'<a href="{url}" style="background-color: {color}; padding: 2px 4px; border-radius: 3px; text-decoration: none; color: black; border: 2px solid #D2691E;" target="_blank" title="{tooltip}">{escaped_entity_text}</a>'
            elif entity.get('wikipedia_url'):
                url = html_module.escape(entity["wikipedia_url"])
                replacement = f'<a href="{url}" style="background-color: {color}; padding: 2px 4px; border-radius: 3px; text-decoration: none; color: black;" target="_blank" title="{tooltip}">{escaped_entity_text}</a>'
            elif entity.get('wikidata_url'):
                url = html_module.escape(entity["wikidata_url"])
                replacement = f'<a href="{url}" style="background-color: {color}; padding: 2px 4px; border-radius: 3px; text-decoration: none; color: black;" target="_blank" title="{tooltip}">{escaped_entity_text}</a>'
            elif entity.get('britannica_url'):
                url = html_module.escape(entity["britannica_url"])
                replacement = f'<a href="{url}" style="background-color: {color}; padding: 2px 4px; border-radius: 3px; text-decoration: none; color: black;" target="_blank" title="{tooltip}">{escaped_entity_text}</a>'
            elif entity.get('openstreetmap_url'):
                url = html_module.escape(entity["openstreetmap_url"])
                replacement = f'<a href="{url}" style="background-color: {color}; padding: 2px 4px; border-radius: 3px; text-decoration: none; color: black;" target="_blank" title="{tooltip}">{escaped_entity_text}</a>'
            else:
                # Just highlight with coordinates (no link)
                replacement = f'<span style="background-color: {color}; padding: 2px 4px; border-radius: 3px;" title="{tooltip}">{escaped_entity_text}</span>'
            
            # Calculate positions in escaped text
            text_before_entity = html_module.escape(text[:start])
            text_entity_escaped = html_module.escape(text[start:end])
            
            escaped_start = len(text_before_entity)
            escaped_end = escaped_start + len(text_entity_escaped)
            
            # Replace in the escaped text
            highlighted = highlighted[:escaped_start] + replacement + highlighted[escaped_end:]
        
        return highlighted

    def render_results(self):
        """Render the results section with entities and visualizations."""
        if not st.session_state.entities:
            st.info("Enter some text above and click 'Process Text' to see results.")
            return
        
        entities = st.session_state.entities
        
        st.header("Results")
        
        # Statistics
        self.render_statistics(entities)
        
        # Highlighted text
        st.subheader("Highlighted Text")
        st.markdown("Entities are highlighted and linked. Places with Pelagios data have special borders:")
        
        # Legend
        col1, col2, col3 = st.columns(3)
        with col1:
            st.markdown("**Pleiades** (brown border)")
        with col2:
            st.markdown("**Peripleo** (orange border)")
        with col3:
            st.markdown("**Other sources** (no border)")
        
        if st.session_state.html_content:
            st.markdown(
                st.session_state.html_content,
                unsafe_allow_html=True
            )
        else:
            st.info("No highlighted text available. Process some text first.")
        
        # Pelagios enhanced entities section
        pelagios_entities = [e for e in entities if e.get('pelagios_data')]
        if pelagios_entities:
            with st.expander(f"Pelagios Enhanced Places ({len(pelagios_entities)})", expanded=True):
                for entity in pelagios_entities:
                    pelagios_data = entity['pelagios_data']
                    col1, col2 = st.columns([2, 3])
                    
                    with col1:
                        st.write(f"**{entity['text']}**")
                        if entity.get('pleiades_url'):
                            st.markdown(f"[Pleiades]({entity['pleiades_url']})")
                        st.markdown(f"[Peripleo]({pelagios_data.get('peripleo_url', '#')})")
                    
                    with col2:
                        if pelagios_data.get('peripleo_description'):
                            st.write(pelagios_data['peripleo_description'])
                        if pelagios_data.get('temporal_bounds'):
                            st.write(f"**Period:** {pelagios_data['temporal_bounds']}")
                        if pelagios_data.get('place_types'):
                            st.write(f"**Types:** {', '.join(pelagios_data['place_types'])}")
                    
                    st.write("---")
        
        # Maps section
        geo_entities = [e for e in entities if e.get('latitude') and e.get('longitude')]
        if geo_entities:
            st.subheader("Geographic Visualization")
            
            # Create map
            self.render_map(geo_entities)
            
            # Peripleo map link
            if pelagios_entities:
                map_url = self.pelagios.create_pelagios_map_url(entities)
                st.markdown(f"[View all places on Peripleo Map]({map_url})")
        
        # Entity details in collapsible section for mobile
        with st.expander("Entity Details", expanded=False):
            self.render_entity_table(entities)
        
        # Export options in collapsible section for mobile
        with st.expander("Export Results", expanded=False):
            self.render_export_section(entities)

    def render_statistics(self, entities: List[Dict[str, Any]]):
        """Render statistics about the extracted entities."""
        # Create columns for metrics (works well on mobile)
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Total Entities", len(entities))
        
        with col2:
            geocoded_count = len([e for e in entities if e.get('latitude')])
            st.metric("Geocoded Places", geocoded_count)
        
        with col3:
            linked_count = len([e for e in entities if e.get('wikidata_url') or e.get('wikipedia_url') or e.get('britannica_url')])
            st.metric("Linked Entities", linked_count)
        
        with col4:
            pelagios_count = len([e for e in entities if e.get('pelagios_data')])
            st.metric("Pelagios Enhanced", pelagios_count)

    def render_map(self, geo_entities: List[Dict[str, Any]]):
        """Render an interactive map of geocoded entities."""
        if not geo_entities:
            return
            
        # Create DataFrame for plotting
        map_data = []
        for entity in geo_entities:
            # Determine source and color
            if entity.get('pleiades_url'):
                source = "Pleiades"
                color = "brown"
            elif entity.get('pelagios_data'):
                source = "Peripleo"
                color = "orange"
            elif entity.get('geocoding_source', '').startswith('geopy'):
                source = "Geopy"
                color = "blue"
            elif entity.get('geocoding_source', '').startswith('openstreetmap'):
                source = "OpenStreetMap"
                color = "green"
            else:
                source = "Other"
                color = "gray"
            
            map_data.append({
                'Entity': entity['text'],
                'Type': entity['type'],
                'Latitude': entity['latitude'],
                'Longitude': entity['longitude'],
                'Source': source,
                'Color': color,
                'Description': entity.get('location_name', ''),
                'Pelagios': 'Yes' if entity.get('pelagios_data') else 'No'
            })
        
        df_map = pd.DataFrame(map_data)
        
        # Create Plotly map
        fig = px.scatter_mapbox(
            df_map,
            lat="Latitude",
            lon="Longitude",
            hover_name="Entity",
            hover_data=["Type", "Source", "Pelagios"],
            color="Source",
            size_max=15,
            zoom=2,
            height=500,
            title="Geographic Distribution of Entities"
        )
        
        fig.update_layout(
            mapbox_style="open-street-map",
            margin={"r": 0, "t": 30, "l": 0, "b": 0}
        )
        
        st.plotly_chart(fig, use_container_width=True)

    def render_entity_table(self, entities: List[Dict[str, Any]]):
        """Render a table of entity details."""
        if not entities:
            st.info("No entities found.")
            return
        
        # Prepare data for table
        table_data = []
        for entity in entities:
            row = {
                'Entity': entity['text'],
                'Type': entity['type'],
                'Links': self.format_entity_links(entity),
                'Pelagios': 'Yes' if entity.get('pelagios_data') else 'No'
            }
            
            # Add description from various sources
            if entity.get('pelagios_data', {}).get('peripleo_description'):
                row['Description'] = entity['pelagios_data']['peripleo_description']
            elif entity.get('wikidata_description'):
                row['Description'] = entity['wikidata_description']
            elif entity.get('wikipedia_description'):
                row['Description'] = entity['wikipedia_description']
            elif entity.get('britannica_title'):
                row['Description'] = entity['britannica_title']
            
            if entity.get('latitude'):
                row['Coordinates'] = f"{entity['latitude']:.4f}, {entity['longitude']:.4f}"
                row['Location'] = entity.get('location_name', '')
            
            # Add temporal bounds if available
            if entity.get('pelagios_data', {}).get('temporal_bounds'):
                row['Period'] = entity['pelagios_data']['temporal_bounds']
            
            table_data.append(row)
        
        # Create DataFrame and display
        df = pd.DataFrame(table_data)
        st.dataframe(df, use_container_width=True)

    def format_entity_links(self, entity: Dict[str, Any]) -> str:
        """Format entity links for display in table."""
        links = []
        if entity.get('pleiades_url'):
            links.append("Pleiades")
        if entity.get('pelagios_data', {}).get('peripleo_url'):
            links.append("Peripleo")
        if entity.get('wikipedia_url'):
            links.append("Wikipedia")
        if entity.get('wikidata_url'):
            links.append("Wikidata")
        if entity.get('britannica_url'):
            links.append("Britannica")
        if entity.get('openstreetmap_url'):
            links.append("OpenStreetMap")
        return " | ".join(links) if links else "No links"

    def render_export_section(self, entities: List[Dict[str, Any]]):
        """Render export options for the results."""
        st.subheader("Pelagios Integration Exports")
        
        # Pelagios-specific exports
        col1, col2, col3 = st.columns(3)
        
        with col1:
            # Recogito JSON export
            recogito_json = self.pelagios.export_to_recogito_format(
                st.session_state.processed_text, 
                entities, 
                st.session_state.analysis_title
            )
            st.download_button(
                label="Download for Recogito",
                data=recogito_json,
                file_name=f"{st.session_state.analysis_title}_recogito.json",
                mime="application/json",
                help="Import this file into Recogito for collaborative annotation",
                use_container_width=True
            )
        
        with col2:
            # TEI XML export
            tei_xml = self.pelagios.export_to_tei_xml(
                st.session_state.processed_text, 
                entities, 
                st.session_state.analysis_title
            )
            st.download_button(
                label="Download TEI XML",
                data=tei_xml,
                file_name=f"{st.session_state.analysis_title}_tei.xml",
                mime="application/xml",
                help="TEI XML with place name markup for digital humanities",
                use_container_width=True
            )
        
        with col3:
            # Standard JSON-LD export
            json_data = self.create_jsonld_export(entities)
            json_str = json.dumps(json_data, indent=2, ensure_ascii=False)
            
            st.download_button(
                label="Download JSON-LD",
                data=json_str,
                file_name=f"{st.session_state.analysis_title}_entities.jsonld",
                mime="application/ld+json",
                help="Structured linked data export",
                use_container_width=True
            )
        
        # Standard exports
        st.subheader("Standard Exports")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # HTML export
            if st.session_state.html_content:
                html_template = self.create_html_export(entities)
                
                st.download_button(
                    label="Download HTML",
                    data=html_template,
                    file_name=f"{st.session_state.analysis_title}_entities.html",
                    mime="text/html",
                    help="Standalone HTML file with highlighted entities",
                    use_container_width=True
                )
        
        with col2:
            # CSV export for analysis
            csv_data = self.create_csv_export(entities)
            
            st.download_button(
                label="Download CSV",
                data=csv_data,
                file_name=f"{st.session_state.analysis_title}_entities.csv",
                mime="text/csv",
                help="CSV file for analysis in spreadsheet applications",
                use_container_width=True
            )

    def create_jsonld_export(self, entities: List[Dict[str, Any]]) -> Dict:
        """Create JSON-LD export format."""
        json_data = {
            "@context": "http://schema.org/",
            "@type": "TextDigitalDocument",
            "text": st.session_state.processed_text,
            "dateCreated": str(pd.Timestamp.now().isoformat()),
            "title": st.session_state.analysis_title,
            "entities": []
        }
        
        # Format entities for JSON-LD
        for entity in entities:
            entity_data = {
                "name": entity['text'],
                "type": entity['type'],
                "startOffset": entity['start'],
                "endOffset": entity['end']
            }
            
            # Add Pelagios data
            if entity.get('pelagios_data'):
                pelagios_data = entity['pelagios_data']
                entity_data['pelagios'] = {
                    "peripleo_id": pelagios_data.get('peripleo_id'),
                    "peripleo_url": pelagios_data.get('peripleo_url'),
                    "temporal_bounds": pelagios_data.get('temporal_bounds'),
                    "place_types": pelagios_data.get('place_types', [])
                }
            
            # Add various links
            same_as = []
            if entity.get('pleiades_url'):
                same_as.append(entity['pleiades_url'])
            if entity.get('wikidata_url'):
                same_as.append(entity['wikidata_url'])
            if entity.get('wikipedia_url'):
                same_as.append(entity['wikipedia_url'])
            if entity.get('britannica_url'):
                same_as.append(entity['britannica_url'])
            
            if same_as:
                entity_data['sameAs'] = same_as if len(same_as) > 1 else same_as[0]
            
            # Add description
            if entity.get('pelagios_data', {}).get('peripleo_description'):
                entity_data['description'] = entity['pelagios_data']['peripleo_description']
            elif entity.get('wikidata_description'):
                entity_data['description'] = entity['wikidata_description']
            elif entity.get('wikipedia_description'):
                entity_data['description'] = entity['wikipedia_description']
            
            # Add coordinates
            if entity.get('latitude') and entity.get('longitude'):
                entity_data['geo'] = {
                    "@type": "GeoCoordinates",
                    "latitude": entity['latitude'],
                    "longitude": entity['longitude']
                }
                if entity.get('location_name'):
                    entity_data['geo']['name'] = entity['location_name']
            
            json_data['entities'].append(entity_data)
        
        return json_data

    def create_html_export(self, entities: List[Dict[str, Any]]) -> str:
        """Create standalone HTML export."""
        html_template = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>Entity Analysis: {st.session_state.analysis_title}</title>
            <meta charset="utf-8">
            <meta name="viewport" content="width=device-width, initial-scale=1">
            <style>
                body {{ font-family: Arial, sans-serif; max-width: 800px; margin: 0 auto; padding: 20px; }}
                .content {{ background: white; padding: 20px; border: 1px solid #ddd; border-radius: 5px; line-height: 1.6; }}
                .header {{ background: #f5f5f5; padding: 15px; border-radius: 5px; margin-bottom: 20px; }}
                .pelagios-entity {{ border: 2px solid #D2691E; }}
                .pleiades-entity {{ border: 2px solid #8B4513; }}
                .statistics {{ background: #f9f9f9; padding: 15px; border-radius: 5px; margin-bottom: 20px; }}
                @media (max-width: 768px) {{
                    body {{ padding: 10px; }}
                    .content {{ padding: 15px; }}
                    .header {{ padding: 10px; }}
                }}
            </style>
        </head>
        <body>
            <div class="header">
                <h1>Entity Analysis: {st.session_state.analysis_title}</h1>
                <p>Generated on {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
            </div>
            <div class="statistics">
                <h3>Statistics</h3>
                <p><strong>Total entities:</strong> {len(entities)}</p>
                <p><strong>Pelagios enhanced:</strong> {len([e for e in entities if e.get('pelagios_data')])}</p>
                <p><strong>Geocoded places:</strong> {len([e for e in entities if e.get('latitude')])}</p>
            </div>
            <div class="content">
                {st.session_state.html_content}
            </div>
        </body>
        </html>
        """
        
        return html_template

    def create_csv_export(self, entities: List[Dict[str, Any]]) -> str:
        """Create CSV export for analysis."""
        import io
        
        # Prepare data for CSV
        csv_data = []
        for entity in entities:
            row = {
                'Entity': entity['text'],
                'Type': entity['type'],
                'Start': entity['start'],
                'End': entity['end'],
                'Pelagios_Enhanced': 'Yes' if entity.get('pelagios_data') else 'No',
                'Has_Coordinates': 'Yes' if entity.get('latitude') else 'No',
                'Latitude': entity.get('latitude', ''),
                'Longitude': entity.get('longitude', ''),
                'Wikidata_URL': entity.get('wikidata_url', ''),
                'Wikipedia_URL': entity.get('wikipedia_url', ''),
                'Pleiades_URL': entity.get('pleiades_url', ''),
                'Peripleo_URL': entity.get('pelagios_data', {}).get('peripleo_url', ''),
                'Temporal_Bounds': entity.get('pelagios_data', {}).get('temporal_bounds', ''),
                'Description': (entity.get('pelagios_data', {}).get('peripleo_description') or 
                              entity.get('wikidata_description') or 
                              entity.get('wikipedia_description') or '')
            }
            csv_data.append(row)
        
        # Convert to CSV
        df = pd.DataFrame(csv_data)
        return df.to_csv(index=False)

    def run(self):
        """Main application runner."""
        # Add custom CSS for Farrow & Ball Slipper Satin background
        st.markdown("""
        <style>
        .stApp {
            background-color: #F5F0DC !important;
        }
        .main .block-container {
            background-color: #F5F0DC !important;
        }
        .stSidebar {
            background-color: #F5F0DC !important;
        }
        .stSelectbox > div > div {
            background-color: white !important;
        }
        .stTextInput > div > div > input {
            background-color: white !important;
        }
        .stTextArea > div > div > textarea {
            background-color: white !important;
        }
        .stExpander {
            background-color: white !important;
            border: 1px solid #E0D7C0 !important;
            border-radius: 4px !important;
        }
        .stDataFrame {
            background-color: white !important;
        }
        </style>
        """, unsafe_allow_html=True)
        
        # Render header
        self.render_header()
        
        # Render sidebar
        self.render_sidebar()
        
        # Single column layout for mobile compatibility
        # Input section
        text_input, analysis_title = self.render_input_section()
        
        # Process button with custom Farrow & Ball Dead Salmon color
        st.markdown("""
        <style>
        .stButton > button {
            background-color: #C4A998 !important;
            color: black !important;
            border: none !important;
            border-radius: 4px !important;
            font-weight: 500 !important;
        }
        .stButton > button:hover {
            background-color: #B5998A !important;
            color: black !important;
        }
        .stButton > button:active {
            background-color: #A68977 !important;
            color: black !important;
        }
        </style>
        """, unsafe_allow_html=True)
        
        if st.button("Process Text", type="primary", use_container_width=True):
            if text_input.strip():
                self.process_text(text_input, analysis_title)
            else:
                st.warning("Please enter some text to analyze.")
        
        # Add some spacing
        st.markdown("---")
        
        # Results section
        self.render_results()


def main():
    """Main function to run the Streamlit application."""
    app = StreamlitEntityLinker()
    app.run()


if __name__ == "__main__":
    main()
