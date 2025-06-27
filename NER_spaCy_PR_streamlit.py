# G#!/usr/bin/env python3
"""
Streamlit Entity Linker Application with Enhanced Pelagios Integration

A web interface for Named Entity Recognition using spaCy with proper
gazetteer hierarchy: Pelagios/Pleiades → GeoNames → OpenStreetMap → Wikidata

Author: EntityLinker Team
Version: 2.1 - Enhanced Gazetteer Hierarchy
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
    """Integration class for Pelagios services with proper hierarchy."""
    
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
            headers = {'User-Agent': 'EntityLinker-Pelagios/2.1'}
            
            response = requests.get(url, params=params, headers=headers, timeout=10)
            if response.status_code == 200:
                data = response.json()
                return data.get('items', [])
            
        except Exception as e:
            print(f"Peripleo search failed for {place_name}: {e}")
            return []
        
        return []
    
    def search_pleiades(self, place_name: str) -> Optional[Dict]:
        """
        Search Pleiades gazetteer directly.
        
        Args:
            place_name: Name of place to search
            
        Returns:
            Pleiades place record if found
        """
        try:
            # Pleiades search API
            url = f"{self.pleiades_base_url}/places/search"
            params = {
                'q': place_name,
                'limit': 1
            }
            headers = {'User-Agent': 'EntityLinker-Pelagios/2.1'}
            
            response = requests.get(url, params=params, headers=headers, timeout=10)
            if response.status_code == 200:
                data = response.json()
                if data.get('features'):
                    return data['features'][0]
            
        except Exception as e:
            print(f"Pleiades search failed for {place_name}: {e}")
        
        return None
    
    def enhance_entities_with_pelagios(self, entities: List[Dict]) -> List[Dict]:
        """
        Enhance place entities with Pelagios data - FIRST PRIORITY.
        
        Args:
            entities: List of extracted entities
            
        Returns:
            Enhanced entities with Pelagios links
        """
        place_types = ['GPE', 'LOCATION', 'FACILITY']
        
        for entity in entities:
            if entity['type'] in place_types:
                # First try Pleiades directly
                pleiades_result = self.search_pleiades(entity['text'])
                if pleiades_result:
                    properties = pleiades_result.get('properties', {})
                    geometry = pleiades_result.get('geometry')
                    
                    # Add Pleiades data
                    entity['pleiades_id'] = properties.get('id')
                    entity['pleiades_url'] = f"{self.pleiades_base_url}/places/{properties.get('id')}"
                    entity['pleiades_title'] = properties.get('title')
                    entity['pleiades_description'] = properties.get('description')
                    
                    # Extract coordinates from Pleiades
                    if geometry and geometry.get('coordinates'):
                        coords = geometry['coordinates']
                        if len(coords) >= 2:
                            entity['longitude'] = coords[0]
                            entity['latitude'] = coords[1]
                            entity['geocoding_source'] = 'pleiades'
                    
                    # Mark as Pelagios enhanced
                    entity['pelagios_data'] = {
                        'source': 'pleiades',
                        'pleiades_id': properties.get('id'),
                        'title': properties.get('title'),
                        'description': properties.get('description')
                    }
                    
                    time.sleep(0.2)  # Rate limiting
                    continue
                
                # If not in Pleiades, try Peripleo
                peripleo_results = self.search_peripleo(entity['text'])
                
                if peripleo_results:
                    best_match = peripleo_results[0]  # Take first result
                    
                    # Add Pelagios data to entity
                    entity['pelagios_data'] = {
                        'source': 'peripleo',
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
                    
                    # Check if this is linked to Pleiades
                    if best_match.get('source_gazetteer') == 'pleiades':
                        pleiades_id = best_match.get('source_id')
                        if pleiades_id:
                            entity['pleiades_id'] = pleiades_id
                            entity['pleiades_url'] = f"{self.pleiades_base_url}/places/{pleiades_id}"
                
                time.sleep(0.2)  # Rate limiting
        
        return entities
    
    def search_geonames(self, place_name: str) -> Optional[Dict]:
        """
        Search GeoNames gazetteer - SECOND PRIORITY.
        
        Args:
            place_name: Name of place to search
            
        Returns:
            GeoNames place record if found
        """
        try:
            url = "http://api.geonames.org/searchJSON"
            params = {
                'q': place_name,
                'maxRows': 1,
                'username': 'demo',  # You should register for your own username
                'style': 'full'
            }
            
            response = requests.get(url, params=params, timeout=10)
            if response.status_code == 200:
                data = response.json()
                geonames = data.get('geonames', [])
                if geonames:
                    return geonames[0]
            
        except Exception as e:
            print(f"GeoNames search failed for {place_name}: {e}")
        
        return None
    
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
                
                # Add place identification - PRIORITIZE PELAGIOS!
                if entity.get('pleiades_url'):
                    annotation["body"].append({
                        "type": "SpecificResource",
                        "purpose": "identifying",
                        "source": {
                            "id": entity['pleiades_url'],
                            "label": entity['text']
                        }
                    })
                elif entity.get('pelagios_data', {}).get('peripleo_url'):
                    annotation["body"].append({
                        "type": "SpecificResource",
                        "purpose": "identifying",
                        "source": {
                            "id": entity['pelagios_data']['peripleo_url'],
                            "label": entity['text']
                        }
                    })
                elif entity.get('geonames_url'):
                    annotation["body"].append({
                        "type": "SpecificResource", 
                        "purpose": "identifying",
                        "source": {
                            "id": entity['geonames_url'],
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
            
            # Add attributes - PRIORITIZE PELAGIOS REFERENCES!
            if entity.get('pleiades_url'):
                place_elem.set('ref', entity['pleiades_url'])
            elif entity.get('pelagios_data', {}).get('peripleo_url'):
                place_elem.set('ref', entity['pelagios_data']['peripleo_url'])
            elif entity.get('geonames_url'):
                place_elem.set('ref', entity['geonames_url'])
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
    Enhanced Entity linking with proper gazetteer hierarchy:
    1. Pelagios/Pleiades (historical places - HIGHEST PRIORITY)
    2. GeoNames (comprehensive global gazetteer)  
    3. OpenStreetMap (community-maintained, current)
    4. Wikidata (LAST RESORT ONLY)
    
    NO Wikipedia linking at all.
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

    def get_coordinates_with_hierarchy(self, entities):
        """
        Enhanced coordinate lookup following the proper gazetteer hierarchy:
        1. Skip if already has Pelagios data (already handled)
        2. Try GeoNames
        3. Try OpenStreetMap  
        4. Try aggressive OpenStreetMap variations
        """
        import requests
        import time
        
        place_types = ['GPE', 'LOCATION', 'FACILITY', 'ORGANIZATION', 'ADDRESS']
        
        for entity in entities:
            if entity['type'] in place_types:
                # Skip if already has coordinates from Pelagios
                if entity.get('latitude') is not None:
                    continue
                    
                # Try GeoNames FIRST (after Pelagios)
                if self._try_geonames_geocoding(entity):
                    continue
                    
                # Try OpenStreetMap
                if self._try_openstreetmap(entity):
                    continue
                    
                # Try aggressive OpenStreetMap with variations
                self._try_aggressive_openstreetmap(entity)
        
        return entities
    
    def _try_geonames_geocoding(self, entity):
        """Try GeoNames geocoding - SECOND PRIORITY after Pelagios."""
        try:
            url = "http://api.geonames.org/searchJSON"
            params = {
                'q': entity['text'],
                'maxRows': 1,
                'username': 'demo',  # You should register for your own username
                'style': 'full'
            }
            
            response = requests.get(url, params=params, timeout=10)
            if response.status_code == 200:
                data = response.json()
                geonames = data.get('geonames', [])
                if geonames:
                    result = geonames[0]
                    entity['latitude'] = float(result['lat'])
                    entity['longitude'] = float(result['lng'])
                    entity['location_name'] = result.get('name', '')
                    entity['geocoding_source'] = 'geonames'
                    entity['geonames_id'] = result.get('geonameId')
                    entity['geonames_url'] = f"http://www.geonames.org/{result.get('geonameId')}"
                    if result.get('countryName'):
                        entity['country'] = result['countryName']
                    return True
        
            time.sleep(0.3)  # Rate limiting
        except Exception as e:
            print(f"GeoNames geocoding failed for {entity['text']}: {e}")
            pass
        
        return False
    
    def _try_openstreetmap(self, entity):
        """Try OpenStreetMap Nominatim API - THIRD PRIORITY."""
        try:
            url = "https://nominatim.openstreetmap.org/search"
            params = {
                'q': entity['text'],
                'format': 'json',
                'limit': 1,
                'addressdetails': 1
            }
            headers = {'User-Agent': 'EntityLinker/2.1'}
        
            response = requests.get(url, params=params, headers=headers, timeout=10)
            if response.status_code == 200:
                data = response.json()
                if data:
                    result = data[0]
                    entity['latitude'] = float(result['lat'])
                    entity['longitude'] = float(result['lon'])
                    entity['location_name'] = result['display_name']
                    entity['geocoding_source'] = 'openstreetmap'
                    entity['osm_id'] = result.get('osm_id')
                    entity['osm_type'] = result.get('osm_type')
                    return True
        
            time.sleep(0.3)  # Rate limiting
        except Exception as e:
            print(f"OpenStreetMap geocoding failed for {entity['text']}: {e}")
            pass
        
        return False
    
    def _try_aggressive_openstreetmap(self, entity):
        """Try more aggressive OpenStreetMap geocoding with variations."""
        import requests
        import time
        
        # Try variations of the entity name
        search_variations = [
            entity['text'],
            f"{entity['text']}, UK",
            f"{entity['text']}, England", 
            f"{entity['text']}, Scotland",
            f"{entity['text']}, Wales",
            f"{entity['text']} city",
            f"{entity['text']} town",
            f"{entity['text']}, United States",
            f"{entity['text']}, USA"
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
                headers = {'User-Agent': 'EntityLinker/2.1'}
            
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
                        entity['osm_id'] = result.get('osm_id')
                        entity['osm_type'] = result.get('osm_type')
                        return True
            
                time.sleep(0.2)  # Rate limiting between attempts
            except Exception:
                continue
        
        return False

    def link_to_wikidata_fallback(self, entities):
        """Add Wikidata linking - ONLY as FINAL FALLBACK for entities without other links."""
        import requests
        import time
        
        for entity in entities:
            # SKIP if already has Pelagios, GeoNames, or OpenStreetMap data
            if (entity.get('pelagios_data') or 
                entity.get('geonames_url') or 
                entity.get('osm_id')):
                continue
                
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
                        entity['wikidata_id'] = result['id']
                
                time.sleep(0.1)  # Rate limiting
            except Exception:
                pass  # Continue if API call fails
        
        return entities


class StreamlitEntityLinker:
    """
    Streamlit wrapper for the EntityLinker class with enhanced Pelagios integration.
    
    Provides a web interface with proper gazetteer hierarchy visualization.
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
    def cached_enhance_with_pelagios(_self, entities_json: str) -> str:
        """Cached Pelagios enhancement."""
        import json
        entities = json.loads(entities_json)
        enhanced_entities = _self.pelagios.enhance_entities_with_pelagios(entities)
        return json.dumps(enhanced_entities, default=str)

    @st.cache_data  
    def cached_link_to_wikidata(_self, entities_json: str) -> str:
        """Cached Wikidata linking - FINAL FALLBACK ONLY."""
        import json
        entities = json.loads(entities_json)
        linked_entities = _self.entity_linker.link_to_wikidata_fallback(entities)
        return json.dumps(linked_entities, default=str)

    def render_header(self):
        """Render the application header with logo."""
        # Display logo if it exists
        try:
            logo_path = "logo.png"
            if os.path.exists(logo_path):
                st.image(logo_path, width=300)
            else:
                st.info("Place your logo.png file in the same directory as this app to display it here")
        except Exception as e:
            st.warning(f"Could not load logo: {e}")        
        
        st.markdown("<br>", unsafe_allow_html=True)
        
        # Main title and description
        st.header("From Text to Linked Data using spaCy + Pelagios")
        st.markdown("**Extract and link named entities using the proper gazetteer hierarchy: Pelagios → GeoNames → OpenStreetMap → Wikidata**")
        
        # Process diagram showing the hierarchy
        st.markdown("""
        <div style="background-color: white; padding: 20px; border-radius: 10px; margin: 20px 0; border: 1px solid #E0D7C0;">
            <div style="text-align: center; margin-bottom: 20px;">
                <div style="background-color: #F8F8F8; padding: 10px; border-radius: 5px; display: inline-block; margin: 5px;">
                     <strong>Input Text</strong>
                </div>
                <div style="margin: 10px 0;">↓</div>
                <div style="background-color: #F8F8F8; padding: 10px; border-radius: 5px; display: inline-block; margin: 5px;">
                     <strong>spaCy Entity Recognition</strong>
                </div>
                <div style="margin: 10px 0;">↓</div>
                <div style="text-align: center;">
                    <strong>Gazetteer Hierarchy (Priority Order):</strong>
                </div>
                <div style="margin: 15px 0;">
                    <div style="background-color: #F8F8F8; padding: 8px; border-radius: 5px; display: inline-block; margin: 3px; font-size: 0.9em;">
                         <strong>1. Pelagios/Pleiades</strong><br><small>Historical & ancient places</small>
                    </div>
                    <div style="background-color: #F8F8F8; padding: 8px; border-radius: 5px; display: inline-block; margin: 3px; font-size: 0.9em;">
                        <strong>2. GeoNames</strong><br><small>Global gazetteer</small>
                    </div>
                    <div style="background-color: #F8F8F8; padding: 8px; border-radius: 5px; display: inline-block; margin: 3px; font-size: 0.9em;">
                         <strong>3. OpenStreetMap</strong><br><small>Community data</small>
                    </div>
                    <div style="background-color: #F8F8F8; padding: 8px; border-radius: 5px; display: inline-block; margin: 3px; font-size: 0.9em;">
                         <strong>4. Wikidata</strong><br><small>Last resort only</small>
                    </div>
                </div>
                <div style="margin: 10px 0;">↓</div>
                <div style="text-align: center;">
                    <strong>Export Formats:</strong>
                </div>
                <div style="margin: 15px 0;">
                    <div style="background-color: #F8F8F8; padding: 8px; border-radius: 5px; display: inline-block; margin: 3px; font-size: 0.9em;">
                         <strong>Recogito JSON</strong><br><small>Collaborative annotation</small>
                    </div>
                    <div style="background-color: #F8F8F8; padding: 8px; border-radius: 5px; display: inline-block; margin: 3px; font-size: 0.9em;">
                         <strong>TEI XML</strong><br><small>Digital humanities</small>
                    </div>
                    <div style="background-color: #F8F8F8; padding: 8px; border-radius: 5px; display: inline-block; margin: 3px; font-size: 0.9em;">
                         <strong>JSON-LD</strong><br><small>Linked data format</small>
                    </div>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)

    def render_sidebar(self):
        """Render the sidebar with hierarchy information."""
        st.sidebar.subheader("Gazetteer Hierarchy")
        st.sidebar.markdown("""
        **Priority Order:**
        1. **Pelagios/Pleiades** - Historical places (trusted)
        2. **GeoNames** - Global coverage
        3. **OpenStreetMap** - Community verified  
        4. **Wikidata** - Final fallback only
        
        **No Wikipedia** - Avoided due to inconsistency
        """)
        
        st.sidebar.subheader("Pelagios Integration")
        st.sidebar.info("Historical places are prioritized through Peripleo search and direct Pleiades integration for the most authoritative ancient world geography.")
        
        st.sidebar.subheader("Export Formats")
        st.sidebar.info("Results exported with proper gazetteer attribution for collaborative annotation and linked data applications.")

    def render_input_section(self):
        """Render the text input section."""
        st.header("Input Text")
        
        analysis_title = st.text_input(
            "Analysis Title (optional)",
            placeholder="Enter a title for this analysis...",
            help="This will be used for naming output files"
        )
        
        # Test paragraph from Herodotus for Pelagios testing
        sample_text = """The Persian learned men say that the Phoenicians were the cause of the dispute. These they say came to our seas from the sea which is called Red, and having settled in the country which they still occupy, at once began to make long voyages. Among other places to which they carried Egyptian and Assyrian merchandise, they came to Argos, which was at that time preeminent in every way among the people of what is now called Hellas. The Phoenicians came to Argos, and set out their cargo. On the fifth or sixth day after their arrival, when their wares were almost all sold, many women came to the shore and among them especially the daughter of the king, whose name was Io according to Persians and Greeks alike, the daughter of Inachus. As these stood about the stern of the ship bargaining for the wares they liked, the Phoenicians incited one another to set upon them. Most of the women escaped: Io and others were seized and thrown into the ship, which then sailed away for Egypt."""       
        
        text_input = st.text_area(
            "Enter your text here:",
            value=sample_text,
            height=200,
            placeholder="Paste your text here for entity extraction...",
            help="This sample shows Herodotus' Histories - perfect for testing the Pelagios gazetteer hierarchy!"
        )
        
        with st.expander("Or upload a text file"):
            uploaded_file = st.file_uploader(
                "Choose a text file",
                type=['txt', 'md'],
                help="Upload a plain text file (.txt) or Markdown file (.md)"
            )
            
            if uploaded_file is not None:
                try:
                    uploaded_text = str(uploaded_file.read(), "utf-8")
                    text_input = uploaded_text
                    st.success(f"File uploaded successfully! ({len(uploaded_text)} characters)")
                    if not analysis_title:
                        import os
                        default_title = os.path.splitext(uploaded_file.name)[0]
                        st.session_state.suggested_title = default_title
                except Exception as e:
                    st.error(f"Error reading file: {e}")
        
        if not analysis_title and hasattr(st.session_state, 'suggested_title'):
            analysis_title = st.session_state.suggested_title
        elif not analysis_title and not uploaded_file:
            analysis_title = "herodotus_test"
        
        return text_input, analysis_title or "text_analysis"

    def process_text(self, text: str, title: str):
        """
        Process the input text using the enhanced EntityLinker with proper hierarchy.
        """
        if not text.strip():
            st.warning("Please enter some text to analyze.")
            return
        
        text_hash = hashlib.md5(text.encode()).hexdigest()
        
        if text_hash == st.session_state.last_processed_hash:
            st.info("This text has already been processed. Results shown below.")
            return
        
        with st.spinner("Processing text with gazetteer hierarchy..."):
            try:
                progress_bar = st.progress(0)
                status_text = st.empty()
                
                # Step 1: Extract entities
                status_text.text("Extracting entities with spaCy...")
                progress_bar.progress(15)
                entities = self.cached_extract_entities(text)
                
                # Step 2: PRIORITIZE Pelagios/Pleiades FIRST!
                status_text.text("🏛️ Searching Pelagios & Pleiades (Priority 1)...")
                progress_bar.progress(35)
                entities_json = json.dumps(entities, default=str)
                enhanced_entities_json = self.cached_enhance_with_pelagios(entities_json)
                entities = json.loads(enhanced_entities_json)
                
                # Step 3: Get coordinates using hierarchy (GeoNames → OpenStreetMap)
                status_text.text("🌍 Geocoding with GeoNames & OpenStreetMap...")
                progress_bar.progress(60)
                entities = self.entity_linker.get_coordinates_with_hierarchy(entities)
                
                # Step 4: Wikidata ONLY as final fallback
                status_text.text("📊 Wikidata fallback for remaining entities...")
                progress_bar.progress(80)
                entities_json = json.dumps(entities, default=str)
                final_entities_json = self.cached_link_to_wikidata(entities_json)
                entities = json.loads(final_entities_json)
                
                # Step 5: Generate visualization
                status_text.text("Generating enhanced visualization...")
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
                
                # Show summary with hierarchy breakdown
                pelagios_enhanced = len([e for e in entities if e.get('pelagios_data')])
                geonames_linked = len([e for e in entities if e.get('geonames_url')])
                osm_linked = len([e for e in entities if e.get('osm_id')])
                wikidata_fallback = len([e for e in entities if e.get('wikidata_url')])
                geocoded = len([e for e in entities if e.get('latitude')])
                
                st.success(f"Processing complete! Found {len(entities)} entities")
                
                if pelagios_enhanced > 0:
                    st.info(f"{pelagios_enhanced} places enhanced with Pelagios data (Priority 1)")
                if geonames_linked > 0:
                    st.info(f"{geonames_linked} places linked via GeoNames (Priority 2)")
                if osm_linked > 0:
                    st.info(f"{osm_linked} places linked via OpenStreetMap (Priority 3)")
                if wikidata_fallback > 0:
                    st.warning(f"{wikidata_fallback} places using Wikidata fallback (Priority 4)")
                if geocoded > 0:
                    st.info(f"{geocoded} total places geocoded with coordinates")
                
            except Exception as e:
                st.error(f"Error processing text: {e}")
                st.exception(e)

    def create_highlighted_html(self, text: str, entities: List[Dict[str, Any]]) -> str:
        """
        Create HTML content with highlighted entities showing hierarchy priority.
        """
        import html as html_module
        
        sorted_entities = sorted(entities, key=lambda x: x['start'], reverse=True)
        
        highlighted = html_module.escape(text)
        
        for entity in sorted_entities:
            has_links = (entity.get('pelagios_data') or 
                         entity.get('geonames_url') or 
                         entity.get('osm_id') or
                         entity.get('wikidata_url'))
            has_coordinates = entity.get('latitude') is not None
            
            if not (has_links or has_coordinates):
                continue
                
            start = entity['start']
            end = entity['end']
            original_entity_text = text[start:end]
            escaped_entity_text = html_module.escape(original_entity_text)
            
            # Create tooltip with hierarchy information
            tooltip_parts = [f"Type: {entity['type']}"]
            
            # Show hierarchy source
            if entity.get('pleiades_url'):
                tooltip_parts.append("Source: Pleiades (Priority 1)")
            elif entity.get('pelagios_data'):
                tooltip_parts.append("Source: Peripleo (Priority 1)")
            elif entity.get('geonames_url'):
                tooltip_parts.append("Source: GeoNames (Priority 2)")
            elif entity.get('osm_id'):
                tooltip_parts.append("Source: OpenStreetMap (Priority 3)")
            elif entity.get('wikidata_url'):
                tooltip_parts.append("Source: Wikidata (Priority 4 - Fallback)")
            
            if entity.get('pelagios_data'):
                pelagios_data = entity['pelagios_data']
                if pelagios_data.get('description') or pelagios_data.get('peripleo_description'):
                    desc = pelagios_data.get('description') or pelagios_data.get('peripleo_description')
                    tooltip_parts.append(f"Description: {desc}")
                if pelagios_data.get('temporal_bounds'):
                    tooltip_parts.append(f"Period: {pelagios_data['temporal_bounds']}")
            elif entity.get('wikidata_description'):
                tooltip_parts.append(f"Description: {entity['wikidata_description']}")
            
            if entity.get('location_name'):
                tooltip_parts.append(f"Location: {entity['location_name']}")
            
            tooltip = " | ".join(tooltip_parts)
            
            # Create highlighted span with proper hierarchy links (no borders or colors)
            if entity.get('pleiades_url'):
                url = html_module.escape(entity["pleiades_url"])
                replacement = f'<a href="{url}" style="text-decoration: underline;" target="_blank" title="{tooltip}">{escaped_entity_text}</a>'
            elif entity.get('pelagios_data', {}).get('peripleo_url'):
                url = html_module.escape(entity["pelagios_data"]["peripleo_url"])
                replacement = f'<a href="{url}" style="text-decoration: underline;" target="_blank" title="{tooltip}">{escaped_entity_text}</a>'
            elif entity.get('geonames_url'):
                url = html_module.escape(entity["geonames_url"])
                replacement = f'<a href="{url}" style="text-decoration: underline;" target="_blank" title="{tooltip}">{escaped_entity_text}</a>'
            elif entity.get('osm_id'):
                # Create OpenStreetMap URL
                osm_type = entity.get('osm_type', 'node')
                osm_id = entity['osm_id']
                url = f"https://www.openstreetmap.org/{osm_type}/{osm_id}"
                replacement = f'<a href="{url}" style="text-decoration: underline;" target="_blank" title="{tooltip}">{escaped_entity_text}</a>'
            elif entity.get('wikidata_url'):
                url = html_module.escape(entity["wikidata_url"])
                replacement = f'<a href="{url}" style="text-decoration: underline;" target="_blank" title="{tooltip}">{escaped_entity_text}</a>'
            else:
                # Just add tooltip (no link, no highlighting)
                replacement = f'<span title="{tooltip}">{escaped_entity_text}</span>'
            
            # Calculate positions in escaped text
            text_before_entity = html_module.escape(text[:start])
            text_entity_escaped = html_module.escape(text[start:end])
            
            escaped_start = len(text_before_entity)
            escaped_end = escaped_start + len(text_entity_escaped)
            
            # Replace in the escaped text
            highlighted = highlighted[:escaped_start] + replacement + highlighted[escaped_end:]
        
        return highlighted

    def render_results(self):
        """Render the results section with hierarchy emphasis."""
        if not st.session_state.entities:
            st.info("Enter some text above and click 'Process Text' to see results.")
            return
        
        entities = st.session_state.entities
        
        st.header("Results")
        
        # Enhanced statistics showing hierarchy breakdown
        self.render_hierarchy_statistics(entities)
        
        # Highlighted text
        st.subheader("Highlighted Text")
        st.markdown("**Gazetteer Hierarchy Visualization:**")
        
        # Legend showing hierarchy (no emojis)
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.markdown("**Pleiades** (underlined links)")
        with col2:
            st.markdown("**Peripleo** (underlined links)")
        with col3:
            st.markdown("**GeoNames** (underlined links)")
        with col4:
            st.markdown("**Wikidata** (underlined links)")
        
        st.markdown("**Hierarchy Priority:** Pelagios/Pleiades → GeoNames → OpenStreetMap → Wikidata")
        
        if st.session_state.html_content:
            st.markdown(
                st.session_state.html_content,
                unsafe_allow_html=True
            )
        
        # Hierarchy breakdown sections
        self.render_hierarchy_breakdown(entities)
        
        # Maps section
        geo_entities = [e for e in entities if e.get('latitude') and e.get('longitude')]
        if geo_entities:
            st.subheader("Geographic Visualization")
            self.render_hierarchy_map(geo_entities)
            
            # Peripleo map link
            pelagios_entities = [e for e in entities if e.get('pelagios_data')]
            if pelagios_entities:
                map_url = self.pelagios.create_pelagios_map_url(entities)
                st.markdown(f"[View all places on Peripleo Map]({map_url})")
        
        # Export options
        with st.expander("Export Results", expanded=False):
            self.render_export_section(entities)

    def render_hierarchy_statistics(self, entities: List[Dict[str, Any]]):
        """Render statistics showing gazetteer hierarchy breakdown."""
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            pelagios_count = len([e for e in entities if e.get('pelagios_data')])
            st.metric("Pelagios", pelagios_count, help="Priority 1: Historical places")
        
        with col2:
            geonames_count = len([e for e in entities if e.get('geonames_url')])
            st.metric("GeoNames", geonames_count, help="Priority 2: Global gazetteer")
        
        with col3:
            osm_count = len([e for e in entities if e.get('osm_id')])
            st.metric("OpenStreetMap", osm_count, help="Priority 3: Community data")
        
        with col4:
            wikidata_count = len([e for e in entities if e.get('wikidata_url')])
            st.metric("Wikidata", wikidata_count, help="Priority 4: Final fallback")

    def render_hierarchy_breakdown(self, entities: List[Dict[str, Any]]):
        """Render breakdown by gazetteer hierarchy."""
        
        # Pelagios enhanced entities section
        pelagios_entities = [e for e in entities if e.get('pelagios_data')]
        if pelagios_entities:
            with st.expander(f"Pelagios Enhanced Places ({len(pelagios_entities)}) - Priority 1", expanded=True):
                for entity in pelagios_entities:
                    col1, col2 = st.columns([2, 3])
                    
                    with col1:
                        st.write(f"**{entity['text']}**")
                        if entity.get('pleiades_url'):
                            st.markdown(f"[Pleiades]({entity['pleiades_url']})")
                        if entity.get('pelagios_data', {}).get('peripleo_url'):
                            st.markdown(f"[Peripleo]({entity['pelagios_data']['peripleo_url']})")
                    
                    with col2:
                        pelagios_data = entity['pelagios_data']
                        if pelagios_data.get('description') or pelagios_data.get('peripleo_description'):
                            desc = pelagios_data.get('description') or pelagios_data.get('peripleo_description')
                            st.write(desc)
                        if pelagios_data.get('temporal_bounds'):
                            st.write(f"**Period:** {pelagios_data['temporal_bounds']}")
                        if pelagios_data.get('place_types'):
                            st.write(f"**Types:** {', '.join(pelagios_data['place_types'])}")
                    
                    st.write("---")
        
        # GeoNames entities section
        geonames_entities = [e for e in entities if e.get('geonames_url') and not e.get('pelagios_data')]
        if geonames_entities:
            with st.expander(f"GeoNames Linked Places ({len(geonames_entities)}) - Priority 2", expanded=False):
                for entity in geonames_entities:
                    col1, col2 = st.columns([2, 3])
                    
                    with col1:
                        st.write(f"**{entity['text']}**")
                        st.markdown(f"[GeoNames]({entity['geonames_url']})")
                        if entity.get('country'):
                            st.write(f"Country: {entity['country']}")
                    
                    with col2:
                        if entity.get('location_name'):
                            st.write(f"**Full name:** {entity['location_name']}")
                        if entity.get('latitude') and entity.get('longitude'):
                            st.write(f"**Coordinates:** {entity['latitude']:.4f}, {entity['longitude']:.4f}")
                    
                    st.write("---")
        
        # OpenStreetMap entities section
        osm_entities = [e for e in entities if e.get('osm_id') and not e.get('pelagios_data') and not e.get('geonames_url')]
        if osm_entities:
            with st.expander(f"OpenStreetMap Linked Places ({len(osm_entities)}) - Priority 3", expanded=False):
                for entity in osm_entities:
                    col1, col2 = st.columns([2, 3])
                    
                    with col1:
                        st.write(f"**{entity['text']}**")
                        osm_url = f"https://www.openstreetmap.org/{entity.get('osm_type', 'node')}/{entity['osm_id']}"
                        st.markdown(f"[OpenStreetMap]({osm_url})")
                    
                    with col2:
                        if entity.get('location_name'):
                            st.write(f"**Full name:** {entity['location_name']}")
                        if entity.get('latitude') and entity.get('longitude'):
                            st.write(f"**Coordinates:** {entity['latitude']:.4f}, {entity['longitude']:.4f}")
                        if entity.get('search_term_used'):
                            st.write(f"**Search term:** {entity['search_term_used']}")
                    
                    st.write("---")
        
        # Wikidata fallback entities section
        wikidata_entities = [e for e in entities if e.get('wikidata_url') and not e.get('pelagios_data') and not e.get('geonames_url') and not e.get('osm_id')]
        if wikidata_entities:
            with st.expander(f"Wikidata Fallback ({len(wikidata_entities)}) - Priority 4", expanded=False):
                st.warning("These entities only have Wikidata links - consider manual verification")
                for entity in wikidata_entities:
                    col1, col2 = st.columns([2, 3])
                    
                    with col1:
                        st.write(f"**{entity['text']}**")
                        st.markdown(f"[Wikidata]({entity['wikidata_url']})")
                    
                    with col2:
                        if entity.get('wikidata_description'):
                            st.write(f"**Description:** {entity['wikidata_description']}")
                    
                    st.write("---")
        # GeoNames entities section
        geonames_entities = [e for e in entities if e.get('geonames_url') and not e.get('pelagios_data')]
        if geonames_entities:
            with st.expander(f"🌍 GeoNames Linked Places ({len(geonames_entities)}) - Priority 2", expanded=False):
                for entity in geonames_entities:
                    col1, col2 = st.columns([2, 3])
                    
                    with col1:
                        st.write(f"**{entity['text']}**")
                        st.markdown(f"[GeoNames]({entity['geonames_url']})")
                        if entity.get('country'):
                            st.write(f"Country: {entity['country']}")
                    
                    with col2:
                        if entity.get('location_name'):
                            st.write(f"**Full name:** {entity['location_name']}")
                        if entity.get('latitude') and entity.get('longitude'):
                            st.write(f"**Coordinates:** {entity['latitude']:.4f}, {entity['longitude']:.4f}")
                    
                    st.write("---")
        
        # OpenStreetMap entities section
        osm_entities = [e for e in entities if e.get('osm_id') and not e.get('pelagios_data') and not e.get('geonames_url')]
        if osm_entities:
            with st.expander(f"🗺️ OpenStreetMap Linked Places ({len(osm_entities)}) - Priority 3", expanded=False):
                for entity in osm_entities:
                    col1, col2 = st.columns([2, 3])
                    
                    with col1:
                        st.write(f"**{entity['text']}**")
                        osm_url = f"https://www.openstreetmap.org/{entity.get('osm_type', 'node')}/{entity['osm_id']}"
                        st.markdown(f"[OpenStreetMap]({osm_url})")
                    
                    with col2:
                        if entity.get('location_name'):
                            st.write(f"**Full name:** {entity['location_name']}")
                        if entity.get('latitude') and entity.get('longitude'):
                            st.write(f"**Coordinates:** {entity['latitude']:.4f}, {entity['longitude']:.4f}")
                        if entity.get('search_term_used'):
                            st.write(f"**Search term:** {entity['search_term_used']}")
                    
                    st.write("---")
        
        # Wikidata fallback entities section
        wikidata_entities = [e for e in entities if e.get('wikidata_url') and not e.get('pelagios_data') and not e.get('geonames_url') and not e.get('osm_id')]
        if wikidata_entities:
            with st.expander(f"📊 Wikidata Fallback ({len(wikidata_entities)}) - Priority 4", expanded=False):
                st.warning("These entities only have Wikidata links - consider manual verification")
                for entity in wikidata_entities:
                    col1, col2 = st.columns([2, 3])
                    
                    with col1:
                        st.write(f"**{entity['text']}**")
                        st.markdown(f"[Wikidata]({entity['wikidata_url']})")
                    
                    with col2:
                        if entity.get('wikidata_description'):
                            st.write(f"**Description:** {entity['wikidata_description']}")
                    
                    st.write("---")

    def render_hierarchy_map(self, geo_entities: List[Dict[str, Any]]):
        """Render an interactive map showing hierarchy sources."""
        if not geo_entities:
            return
            
        # Create DataFrame for plotting with hierarchy information
        map_data = []
        for entity in geo_entities:
            # Determine source and styling based on hierarchy
            if entity.get('pleiades_url'):
                source = "Pleiades (Priority 1)"
                color = "#8B4513"  # Brown
                size = 15
            elif entity.get('pelagios_data'):
                source = "Peripleo (Priority 1)"
                color = "#D2691E"  # Orange
                size = 15
            elif entity.get('geonames_url'):
                source = "GeoNames (Priority 2)"
                color = "#4169E1"  # Blue
                size = 12
            elif entity.get('osm_id'):
                source = "OpenStreetMap (Priority 3)"
                color = "#228B22"  # Green
                size = 10
            elif entity.get('wikidata_url'):
                source = "Wikidata (Priority 4)"
                color = "#999999"  # Gray
                size = 8
            else:
                source = "Other"
                color = "#CCCCCC"
                size = 6
            
            map_data.append({
                'Entity': entity['text'],
                'Type': entity['type'],
                'Latitude': entity['latitude'],
                'Longitude': entity['longitude'],
                'Source': source,
                'Color': color,
                'Size': size,
                'Description': entity.get('location_name', ''),
                'Hierarchy_Level': source.split('(')[1].split(')')[0] if '(' in source else 'Unknown'
            })
        
        df_map = pd.DataFrame(map_data)
        
        # Create Plotly map with hierarchy visualization
        fig = px.scatter_mapbox(
            df_map,
            lat="Latitude",
            lon="Longitude",
            hover_name="Entity",
            hover_data=["Type", "Source", "Hierarchy_Level"],
            color="Source",
            size="Size",
            size_max=20,
            zoom=2,
            height=500,
            title="Geographic Distribution by Gazetteer Hierarchy",
            color_discrete_map={
                "Pleiades (Priority 1)": "#8B4513",
                "Peripleo (Priority 1)": "#D2691E", 
                "GeoNames (Priority 2)": "#4169E1",
                "OpenStreetMap (Priority 3)": "#228B22",
                "Wikidata (Priority 4)": "#999999"
            }
        )
        
        fig.update_layout(
            mapbox_style="open-street-map",
            margin={"r": 0, "t": 30, "l": 0, "b": 0}
        )
        
        st.plotly_chart(fig, use_container_width=True)

    def render_export_section(self, entities: List[Dict[str, Any]]):
        """Render export options with hierarchy preservation."""
        st.subheader("Pelagios Integration Exports")
        
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
                help="Import into Recogito with proper gazetteer hierarchy",
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
                help="TEI XML with hierarchical place references",
                use_container_width=True
            )
        
        with col3:
            # Enhanced JSON-LD export
            json_data = self.create_hierarchy_jsonld_export(entities)
            json_str = json.dumps(json_data, indent=2, ensure_ascii=False)
            
            st.download_button(
                label="Download JSON-LD",
                data=json_str,
                file_name=f"{st.session_state.analysis_title}_entities.jsonld",
                mime="application/ld+json",
                help="Linked data with gazetteer hierarchy information",
                use_container_width=True
            )
        
        # Standard exports
        st.subheader("Standard Exports")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Enhanced HTML export
            if st.session_state.html_content:
                html_template = self.create_hierarchy_html_export(entities)
                
                st.download_button(
                    label="Download HTML",
                    data=html_template,
                    file_name=f"{st.session_state.analysis_title}_entities.html",
                    mime="text/html",
                    help="Standalone HTML with hierarchy visualization",
                    use_container_width=True
                )
        
        with col2:
            # Enhanced CSV export
            csv_data = self.create_hierarchy_csv_export(entities)
            
            st.download_button(
                label="Download CSV",
                data=csv_data,
                file_name=f"{st.session_state.analysis_title}_entities.csv",
                mime="text/csv",
                help="CSV with gazetteer hierarchy information",
                use_container_width=True
            )

    def create_hierarchy_jsonld_export(self, entities: List[Dict[str, Any]]) -> Dict:
        """Create JSON-LD export with hierarchy information."""
        json_data = {
            "@context": "http://schema.org/",
            "@type": "TextDigitalDocument",
            "text": st.session_state.processed_text,
            "dateCreated": str(pd.Timestamp.now().isoformat()),
            "title": st.session_state.analysis_title,
            "gazetteerHierarchy": [
                "Pelagios/Pleiades (Priority 1)",
                "GeoNames (Priority 2)", 
                "OpenStreetMap (Priority 3)",
                "Wikidata (Priority 4 - Fallback)"
            ],
            "entities": []
        }
        
        for entity in entities:
            entity_data = {
                "name": entity['text'],
                "type": entity['type'],
                "startOffset": entity['start'],
                "endOffset": entity['end']
            }
            
            # Add hierarchy information
            if entity.get('pelagios_data'):
                entity_data['gazetteerSource'] = "Pelagios (Priority 1)"
                pelagios_data = entity['pelagios_data']
                entity_data['pelagios'] = {
                    "source": pelagios_data.get('source'),
                    "peripleo_id": pelagios_data.get('peripleo_id'),
                    "peripleo_url": pelagios_data.get('peripleo_url'),
                    "temporal_bounds": pelagios_data.get('temporal_bounds'),
                    "place_types": pelagios_data.get('place_types', [])
                }
            elif entity.get('geonames_url'):
                entity_data['gazetteerSource'] = "GeoNames (Priority 2)"
                entity_data['geonames_id'] = entity.get('geonames_id')
            elif entity.get('osm_id'):
                entity_data['gazetteerSource'] = "OpenStreetMap (Priority 3)"
                entity_data['osm_id'] = entity.get('osm_id')
                entity_data['osm_type'] = entity.get('osm_type')
            elif entity.get('wikidata_url'):
                entity_data['gazetteerSource'] = "Wikidata (Priority 4 - Fallback)"
                entity_data['wikidata_id'] = entity.get('wikidata_id')
            
            # Add links in hierarchy order
            same_as = []
            if entity.get('pleiades_url'):
                same_as.append(entity['pleiades_url'])
            if entity.get('pelagios_data', {}).get('peripleo_url'):
                same_as.append(entity['pelagios_data']['peripleo_url'])
            if entity.get('geonames_url'):
                same_as.append(entity['geonames_url'])
            if entity.get('osm_id'):
                osm_url = f"https://www.openstreetmap.org/{entity.get('osm_type', 'node')}/{entity['osm_id']}"
                same_as.append(osm_url)
            if entity.get('wikidata_url'):
                same_as.append(entity['wikidata_url'])
            
            if same_as:
                entity_data['sameAs'] = same_as if len(same_as) > 1 else same_as[0]
            
            # Add description with source priority
            if entity.get('pelagios_data', {}).get('description') or entity.get('pelagios_data', {}).get('peripleo_description'):
                desc = entity['pelagios_data'].get('description') or entity['pelagios_data'].get('peripleo_description')
                entity_data['description'] = desc
            elif entity.get('wikidata_description'):
                entity_data['description'] = entity['wikidata_description']
            
            # Add coordinates
            if entity.get('latitude') and entity.get('longitude'):
                entity_data['geo'] = {
                    "@type": "GeoCoordinates",
                    "latitude": entity['latitude'],
                    "longitude": entity['longitude']
                }
                if entity.get('location_name'):
                    entity_data['geo']['name'] = entity['location_name']
                    
                # Add geocoding source
                if entity.get('geocoding_source'):
                    entity_data['geo']['source'] = entity['geocoding_source']
            
            json_data['entities'].append(entity_data)
        
        return json_data

    def create_hierarchy_html_export(self, entities: List[Dict[str, Any]]) -> str:
        """Create HTML export with hierarchy visualization."""
        # Count by hierarchy
        pelagios_count = len([e for e in entities if e.get('pelagios_data')])
        geonames_count = len([e for e in entities if e.get('geonames_url') and not e.get('pelagios_data')])
        osm_count = len([e for e in entities if e.get('osm_id') and not e.get('pelagios_data') and not e.get('geonames_url')])
        wikidata_count = len([e for e in entities if e.get('wikidata_url') and not e.get('pelagios_data') and not e.get('geonames_url') and not e.get('osm_id')])
        
        html_template = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>Entity Analysis with Gazetteer Hierarchy: {st.session_state.analysis_title}</title>
            <meta charset="utf-8">
            <meta name="viewport" content="width=device-width, initial-scale=1">
            <style>
                body {{ font-family: Arial, sans-serif; max-width: 900px; margin: 0 auto; padding: 20px; background-color: white; }}
                .content {{ background: white; padding: 20px; border: 1px solid #ddd; border-radius: 5px; line-height: 1.6; }}
                .header {{ background: #f5f5f5; padding: 15px; border-radius: 5px; margin-bottom: 20px; }}
                .hierarchy-stats {{ background: #f9f9f9; padding: 15px; border-radius: 5px; margin-bottom: 20px; }}
                .hierarchy-legend {{ background: #fff; padding: 15px; border: 2px solid #ddd; border-radius: 5px; margin-bottom: 20px; }}
                .legend-item {{ display: inline-block; margin: 5px 10px; padding: 5px 10px; border-radius: 3px; }}
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
                <p><strong>Gazetteer Hierarchy:</strong> Pelagios → GeoNames → OpenStreetMap → Wikidata</p>
            </div>
            
            <div class="hierarchy-stats">
                <h3>Gazetteer Hierarchy Breakdown</h3>
                <p><strong>Pelagios/Pleiades (Priority 1):</strong> {pelagios_count} entities</p>
                <p><strong>GeoNames (Priority 2):</strong> {geonames_count} entities</p>
                <p><strong>OpenStreetMap (Priority 3):</strong> {osm_count} entities</p>
                <p><strong>Wikidata (Priority 4):</strong> {wikidata_count} entities</p>
                <p><strong>Total entities:</strong> {len(entities)}</p>
                <p><strong>Geocoded places:</strong> {len([e for e in entities if e.get('latitude')])}</p>
            </div>
            
            <div class="hierarchy-legend">
                <h3>Visual Legend</h3>
                <div class="legend-item">Pelagios/Pleiades (Priority 1)</div>
                <div class="legend-item">GeoNames (Priority 2)</div>
                <div class="legend-item">OpenStreetMap (Priority 3)</div>
                <div class="legend-item">Wikidata (Priority 4 - Fallback)</div>
            </div>
            
            <div class="content">
                {st.session_state.html_content}
            </div>
        </body>
        </html>
        """
        
        return html_template

    def create_hierarchy_csv_export(self, entities: List[Dict[str, Any]]) -> str:
        """Create CSV export with hierarchy information."""
        import io
        
        csv_data = []
        for entity in entities:
            # Determine hierarchy level
            if entity.get('pelagios_data'):
                hierarchy_level = "Priority 1 - Pelagios"
                primary_source = "Pelagios"
            elif entity.get('geonames_url'):
                hierarchy_level = "Priority 2 - GeoNames"
                primary_source = "GeoNames"
            elif entity.get('osm_id'):
                hierarchy_level = "Priority 3 - OpenStreetMap"
                primary_source = "OpenStreetMap"
            elif entity.get('wikidata_url'):
                hierarchy_level = "Priority 4 - Wikidata (Fallback)"
                primary_source = "Wikidata"
            else:
                hierarchy_level = "No Links"
                primary_source = "None"
            
            row = {
                'Entity': entity['text'],
                'Type': entity['type'],
                'Start': entity['start'],
                'End': entity['end'],
                'Hierarchy_Level': hierarchy_level,
                'Primary_Source': primary_source,
                'Has_Coordinates': 'Yes' if entity.get('latitude') else 'No',
                'Latitude': entity.get('latitude', ''),
                'Longitude': entity.get('longitude', ''),
                'Geocoding_Source': entity.get('geocoding_source', ''),
                
                # All possible URLs
                'Pleiades_URL': entity.get('pleiades_url', ''),
                'Peripleo_URL': entity.get('pelagios_data', {}).get('peripleo_url', ''),
                'GeoNames_URL': entity.get('geonames_url', ''),
                'GeoNames_ID': entity.get('geonames_id', ''),
                'OSM_ID': entity.get('osm_id', ''),
                'OSM_Type': entity.get('osm_type', ''),
                'Wikidata_URL': entity.get('wikidata_url', ''),
                'Wikidata_ID': entity.get('wikidata_id', ''),
                
                # Descriptions
                'Pelagios_Description': (entity.get('pelagios_data', {}).get('description') or 
                                       entity.get('pelagios_data', {}).get('peripleo_description') or ''),
                'Temporal_Bounds': entity.get('pelagios_data', {}).get('temporal_bounds', ''),
                'Wikidata_Description': entity.get('wikidata_description', ''),
                'Location_Name': entity.get('location_name', ''),
                'Country': entity.get('country', ''),
                'Search_Term_Used': entity.get('search_term_used', '')
            }
            csv_data.append(row)
        
        df = pd.DataFrame(csv_data)
        return df.to_csv(index=False)

    def run(self):
        """Main application runner with enhanced hierarchy support."""
        # Custom CSS for Farrow & Ball styling
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
        
        self.render_header()
        self.render_sidebar()
        
        # Input section
        text_input, analysis_title = self.render_input_section()
        
        # Enhanced process button styling
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
        
        if st.button("Process Text with Gazetteer Hierarchy", type="primary", use_container_width=True):
            if text_input.strip():
                self.process_text(text_input, analysis_title)
            else:
                st.warning("Please enter some text to analyze.")
        
        st.markdown("---")
        
        # Results section with hierarchy emphasis
        self.render_results()


def main():
    """Main function to run the enhanced Streamlit application."""
    app = StreamlitEntityLinker()
    app.run()


if __name__ == "__main__":
    main()
