debug_container = st.expander("🔧 Processing Details", expanded=st.session_state.get('show_debug', False))
                
                with debug_container:
                    debug_text = st.empty()
                
                # Step 1: Extract entities (cached)
                status_text.text("🔍 Extracting entities with enhanced spaCy...")
                debug_text.text("Starting entity extraction...")
                progress_bar.progress(10)
                entities = self.cached_extract_entities(text)
                debug_text.text(f"Found {len(entities)} initial entities: {[e['text'] for e in entities[:5]]}...")
                
                # Step 1.5: Detect context before entity enhancement
                status_text.text("🔍 Analyzing text context...")
                progress_bar.progress(15)
                context_info = self.entity_linker._detect_geographical_context(text, entities)
                
                if context_info and any(context_info.values()):
                    context_summary = []
                    if context_info.get('regions'):
                        context_summary.append(f"Regions: {', '.join(context_info['regions'])}")
                    if context_info.get('time_period'):
                        context_summary.append(f"Period: {context_info['time_period']}")
                    if context_info.get('ancient_civilizations'):
                        context_summary.append(f"Civilizations: {', '.join(context_info['ancient_civilizations'])}")
                    
                    debug_text.text(f"Detected context: {' | '.join(context_summary)}")
                else:
                    debug_text.text("No specific historical context detected, using general search")
                
                # Store context in session state
                st.session_state.context_info = context_info
                
                # Step 2: Enhance with Pelagios using context
                status_text.text("🏛️ Enhancing with Pelagios historical data (context-aware)...")
                progress_bar.progress(25)
                
                # Use the cached version with context
                entities_json = json.dumps(entities, default=str)
                context_json = json.dumps(context_info, default=str) if context_info else "{}"
                enhanced_entities_json = self.cached_enhance_with_pelagios(entities_json, context_json)
                entities = json.loads(enhanced_entities_json)
                
                pelagios_count = len([e for e in entities if e.get('pelagios_data')])
                debug_text.text(f"Pelagios enhanced {pelagios_count} entities")
                
                # Step 3: Link to Wikidata (cached)
                status_text.text("Linking to Wikidata...")
                progress_bar.progress(45)
                entities_json = json.dumps(entities, default=str)
                linked_entities_json = self.cached_link_to_wikidata(entities_json)
                entities = json.loads(linked_entities_json)
                
                wikidata_count = len([e for e in entities if e.get('wikidata_url')])
                debug_text.text(f"Wikidata linked {wikidata_count} entities")
                
                # Step 4: Link to Wikipedia (cached)
                status_text.text("Linking to Wikipedia...")
                progress_bar.progress(60)
                entities_json = json.dumps(entities, default=str)
                linked_entities_json = self.cached_link_to_wikipedia(entities_json)
                entities = json.loads(linked_entities_json)
                
                wikipedia_count = len([e for e in entities if e.get('wikipedia_url')])
                debug_text.text(f"Wikipedia linked {wikipedia_count} entities")
                
                # Step 5: Link to Britannica (cached)
                status_text.text("Linking to Britannica...")
                progress_bar.progress(75)
                entities_json = json.dumps(entities, default=str)
                linked_entities_json = self.cached_link_to_britannica(entities_json)
                entities = json.loads(linked_entities_json)
                
                britannica_count = len([e for e in entities if e.get('britannica_url')])
                debug_text.text(f"Britannica linked {britannica_count} entities")
                
                # Step 6: Enhanced coordinate lookup
                status_text.text("Getting enhanced coordinates...")
                progress_bar.progress(85)
                entities = self.entity_linker.get_coordinates(entities)
                
                geocoded_count = len([e for e in entities if e.get('latitude')])
                debug_text.text(f"Geocoded {geocoded_count} places")
                
                # Step 7: Generate visualization
                status_text.text("Generating enhanced visualization...")
                progress_bar.progress(95)
                html_content = self.create_highlighted_html(text, entities)
                
                # Store in session state
                st.session_state.entities = entities
                st.session_state.processed_text = text
                st.session_state.html_content = html_content
                st.session_state.analysis_title = title
                st.session_state.last_processed_hash = text_hash
                
                # Complete
                progress_bar.progress(100)
                status_text.empty()
                
                # Enhanced summary
                pelagios_enhanced = len([e for e in entities if e.get('pelagios_data')])
                pleiades_linked = len([e for e in entities if e.get('pleiades_id')])
                total_linked = len([e for e in entities if any([
                    e.get('pleiades_id'), e.get('wikidata_url'), 
                    e.get('wikipedia_url'), e.get('britannica_url')
                ])])
                
                st.success(f"Processing complete! Found {len(entities)} entities")
                
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("Pelagios Enhanced", pelagios_enhanced)
                with col2:
                    st.metric("Pleiades Linked", pleiades_linked)
                with col3:
                    st.metric("Total Linked", total_linked)
                with col4:
                    st.metric("Geocoded", geocoded_count)
                
                if pelagios_enhanced > 0:
                    st.info(f"Successfully enhanced {pelagios_enhanced} places with historical data!")
                
            except Exception as e:
                st.error(f"Error processing text: {e}")
                if st.session_state.get('show_debug', False):
                    st.exception(e)

    def create_highlighted_html(self, text: str, entities: List[Dict[str, Any]]) -> str:
        """
        Enhanced HTML highlighting with improved visual indicators.
        """
        import html as html_module
        
        # Sort entities by start position (reverse for safe replacement)
        sorted_entities = sorted(entities, key=lambda x: x['start'], reverse=True)
        
        # Start with escaped text
        highlighted = html_module.escape(text)
        
        # Enhanced color scheme
        colors = {
            'PERSON': '#BF7B69',          # F&B Red earth        
            'ORGANIZATION': '#9fd2cd',    # F&B Blue ground
            'GPE': '#C4C3A2',             # F&B Cooking apple green
            'LOCATION': '#EFCA89',        # F&B Yellow ground 
            'FACILITY': '#C3B5AC',        # F&B Elephants breath
            'NORP': '#D4C5B9',            # F&B Elephant's breath light
            'ADDRESS': '#CCBEAA'          # F&B Oxford stone
        }
        
        # Replace entities from end to start
        for entity in sorted_entities:
            # Only highlight entities that have links OR coordinates OR Pelagios data
            has_pleiades = entity.get('pleiades_id') is not None
            has_pelagios = entity.get('pelagios_data') is not None
            has_links = any([
                entity.get('wikidata_url'), entity.get('wikipedia_url'), 
                entity.get('britannica_url')
            ])
            has_coordinates = entity.get('latitude') is not None
            
            if not (has_pleiades or has_pelagios or has_links or has_coordinates):
                continue
                
            start = entity['start']
            end = entity['end']
            original_entity_text = text[start:end]
            escaped_entity_text = html_module.escape(original_entity_text)
            color = colors.get(entity['type'], '#E7E2D2')
            
            # Enhanced tooltip with more information
            tooltip_parts = [f"Type: {entity['type']}"]
            
            if entity.get('pelagios_data'):
                pelagios_data = entity['pelagios_data']
                if pelagios_data.get('peripleo_title'):
                    tooltip_parts.append(f"Peripleo: {pelagios_data['peripleo_title']}")
                if pelagios_data.get('temporal_bounds'):
                    tooltip_parts.append(f"Period: {pelagios_data['temporal_bounds']}")
                if pelagios_data.get('place_types'):
                    tooltip_parts.append(f"Types: {', '.join(pelagios_data['place_types'][:2])}")
            
            if entity.get('pleiades_id'):
                tooltip_parts.append(f"Pleiades ID: {entity['pleiades_id']}")
            
            if entity.get('wikidata_description'):
                desc = entity['wikidata_description'][:100] + "..." if len(entity['wikidata_description']) > 100 else entity['wikidata_description']
                tooltip_parts.append(f"Description: {desc}")
            
            if entity.get('location_name'):
                tooltip_parts.append(f"Modern: {entity['location_name'][:50]}...")
            
            if entity.get('geocoding_source'):
                tooltip_parts.append(f"Source: {entity['geocoding_source']}")
            
            tooltip = " | ".join(tooltip_parts)
            
            # Enhanced link priority and visual styling
            border_style = ""
            if has_pleiades:
                url = html_module.escape(entity["pleiades_url"])
                border_style = "border: 2px solid #8B4513; font-weight: bold;"
                replacement = f'<a href="{url}" style="background-color: {color}; padding: 2px 4px; border-radius: 3px; text-decoration: none; color: black; {border_style}" target="_blank" title="{tooltip}">{escaped_entity_text}</a>'
            elif has_pelagios and entity.get('pelagios_data', {}).get('peripleo_url'):
                url = html_module.escape(entity["pelagios_data"]["peripleo_url"])
                border_style = "border: 2px solid #D2691E; font-weight: bold;"
                replacement = f'<a href="{url}" style="background-color: {color}; padding: 2px 4px; border-radius: 3px; text-decoration: none; color: black; {border_style}" target="_blank" title="{tooltip}">{escaped_entity_text}</a>'
            elif entity.get('wikipedia_url'):
                url = html_module.escape(entity["wikipedia_url"])
                replacement = f'<a href="{url}" style="background-color: {color}; padding: 2px 4px; border-radius: 3px; text-decoration: none; color: black;" target="_blank" title="{tooltip}">{escaped_entity_text}</a>'
            elif entity.get('wikidata_url'):
                url = html_module.escape(entity["wikidata_url"])
                replacement = f'<a href="{url}" style="background-color: {color}; padding: 2px 4px; border-radius: 3px; text-decoration: none; color: black;" target="_blank" title="{tooltip}">{escaped_entity_text}</a>'
            elif entity.get('britannica_url'):
                url = html_module.escape(entity["britannica_url"])
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
        """Enhanced results section with better organization."""
        if not st.session_state.entities:
            st.info("Enter some text above and click 'Process Text' to see results.")
            return
        
        entities = st.session_state.entities
        
        st.header("Results")
        
        # Display detected context if available
        if hasattr(st.session_state, 'context_info') and st.session_state.context_info:
            self.render_context_info(st.session_state.context_info)
        
        # Enhanced statistics
        self.render_enhanced_statistics(entities)
        
        # Highlighted text with improved legend
        st.subheader("Highlighted Text")
        
        # Enhanced legend with icons
        st.markdown("""
        **Visual Legend:**
        - **Pleiades entities** (brown border) - Ancient world gazetteer
        - **Peripleo entities** (orange border) - Historical geography
        - **Other linked entities** (colored background) - General knowledge bases
        """)
        
        if st.session_state.html_content:
            # Add custom CSS for better mobile display
            st.markdown("""
            <style>
            .highlighted-text {
                line-height: 1.8;
                font-size: 16px;
                padding: 20px;
                background-color: white;
                border-radius: 10px;
                border: 1px solid #E0D7C0;
                margin: 10px 0;
            }
            @media (max-width: 768px) {
                .highlighted-text {
                    font-size: 14px;
                    padding: 15px;
                    line-height: 1.6;
                }
            }
            </style>
            """, unsafe_allow_html=True)
            
            st.markdown(
                f'<div class="highlighted-text">{st.session_state.html_content}</div>',
                unsafe_allow_html=True
            )
        else:
            st.info("No highlighted text available. Process some text first.")
        
        # Enhanced Pelagios section
        pelagios_entities = [e for e in entities if e.get('pelagios_data') or e.get('pleiades_id')]
        if pelagios_entities:
            with st.expander(f"🏛️ Pelagios & Pleiades Enhanced Places ({len(pelagios_entities)})", expanded=True):
                for entity in pelagios_entities:
                    self.render_pelagios_entity_card(entity)
        
        # Maps section with enhanced features
        geo_entities = [e for e in entities if e.get('latitude') and e.get('longitude')]
        if geo_entities:
            st.subheader("📍 Geographic Visualization")
            
            # Enhanced map
            self.render_enhanced_map(geo_entities)
            
            # Peripleo map link
            if pelagios_entities:
                map_url = self.pelagios.create_pelagios_map_url(entities)
                st.markdown(f"[**View all places on Peripleo Interactive Map**]({map_url})")
        
        # Collapsible sections for mobile
        with st.expander("📊 Entity Details Table", expanded=False):
            self.render_enhanced_entity_table(entities)
        
        with st.expander("💾 Export Results", expanded=False):
            self.render_enhanced_export_section(entities)
    
    def render_context_info(self, context_info: Dict):
        """Display the detected context information."""
        with st.expander("🔍 Detected Historical Context", expanded=True):
            st.markdown("*This context was automatically detected to improve entity linking accuracy*")
            
            cols = st.columns(3)
            
            with cols[0]:
                if context_info.get('regions'):
                    st.markdown("**Geographic Regions:**")
                    for region in context_info['regions']:
                        st.markdown(f"• {region}")
                else:
                    st.markdown("**Geographic Regions:**")
                    st.markdown("*Not detected*")
            
            with cols[1]:
                if context_info.get('time_period'):
                    st.markdown("**Time Period:**")
                    st.markdown(f"• {context_info['time_period']}")
                    if context_info.get('historical_era'):
                        st.markdown(f"• {context_info['historical_era']}")
                else:
                    st.markdown("**Time Period:**")
                    st.markdown("*Not detected*")
            
            with cols[2]:
                if context_info.get('ancient_civilizations'):
                    st.markdown("**Civilizations:**")
                    for civ in context_info['ancient_civilizations']:
                        st.markdown(f"• {civ}")
                else:
                    st.markdown("**Civilizations:**")
                    st.markdown("*Not detected*")
            
            # Show how context improved results
            if any(e.get('context_used') for e in st.session_state.entities):
                st.info("✅ Context was successfully used to disambiguate place names and improve geocoding accuracy.")

    def render_enhanced_statistics(self, entities: List[Dict[str, Any]]):
        """Enhanced statistics with more detailed metrics."""
        # Main metrics
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Total Entities", len(entities))
        
        with col2:
            pelagios_count = len([e for e in entities if e.get('pelagios_data') or e.get('pleiades_id')])
            st.metric("Pelagios Enhanced", pelagios_count)
        
        with col3:
            geocoded_count = len([e for e in entities if e.get('latitude')])
            st.metric("Geocoded", geocoded_count)
        
        with col4:
            linked_count = len([e for e in entities if any([
                e.get('wikidata_url'), e.get('wikipedia_url'), e.get('britannica_url')
            ])])
            st.metric("Linked", linked_count)
        
        # Detailed breakdown
        with st.expander("Detailed Statistics"):
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("Entity Types")
                type_counts = {}
                for entity in entities:
                    entity_type = entity['type']
                    type_counts[entity_type] = type_counts.get(entity_type, 0) + 1
                
                for entity_type, count in sorted(type_counts.items()):
                    st.write(f"**{entity_type}:** {count}")
            
            with col2:
                st.subheader("Linking Success")
                pleiades_count = len([e for e in entities if e.get('pleiades_id')])
                wikidata_count = len([e for e in entities if e.get('wikidata_url')])
                wikipedia_count = len([e for e in entities if e.get('wikipedia_url')])
                britannica_count = len([e for e in entities if e.get('britannica_url')])
                
                st.write(f"**Pleiades:** {pleiades_count}")
                st.write(f"**Wikidata:** {wikidata_count}")
                st.write(f"**Wikipedia:** {wikipedia_count}")
                st.write(f"**Britannica:** {britannica_count}")

    def render_pelagios_entity_card(self, entity: Dict):
        """Render an enhanced card for Pelagios entities."""
        col1, col2 = st.columns([1, 2])
        
        with col1:
            st.markdown(f"### **{entity['text']}**")
            st.write(f"**Type:** {entity['type']}")
            
            # Links with icons
            if entity.get('pleiades_url'):
                st.markdown(f"[Pleiades]({entity['pleiades_url']})")
            
            if entity.get('pelagios_data', {}).get('peripleo_url'):
                st.markdown(f"[Peripleo]({entity['pelagios_data']['peripleo_url']})")
            
            if entity.get('latitude'):
                st.write(f"**Coordinates:** {entity.get('latitude', 'N/A'):.4f}, {entity.get('longitude', 'N/A'):.4f}")
        
        with col2:
            pelagios_data = entity.get('pelagios_data', {})
            
            if pelagios_data.get('peripleo_description'):
                st.write("**Description:**")
                st.write(pelagios_data['peripleo_description'])
            
            if pelagios_data.get('temporal_bounds'):
                st.write(f"**Time Period:** {pelagios_data['temporal_bounds']}")
            
            if pelagios_data.get('place_types'):
                st.write(f"**Place Types:** {', '.join(pelagios_data['place_types'])}")
            
            if entity.get('geocoding_source'):
                st.write(f"**Geocoding Source:** {entity['geocoding_source']}")
        
        st.markdown("---")

    def render_enhanced_map(self, geo_entities: List[Dict[str, Any]]):
        """Enhanced map visualization with better categorization."""
        if not geo_entities:
            return
            
        # Create enhanced DataFrame for plotting
        map_data = []
        for entity in geo_entities:
            # Determine source and styling
            if entity.get('pleiades_url'):
                source = "Pleiades"
                color = "#8B4513"
                symbol = "circle"
                size = 12
            elif entity.get('pelagios_data'):
                source = "Peripleo"
                color = "#D2691E"
                symbol = "square"
                size = 10
            elif entity.get('geocoding_source', '').startswith('geopy'):
                source = "Geopy"
                color = "#4169E1"
                symbol = "triangle-up"
                size = 8
            elif entity.get('geocoding_source', '').startswith('openstreetmap'):
                source = "OpenStreetMap"
                color = "#228B22"
                symbol = "diamond"
                size = 8
            else:
                source = "Other"
                color = "#808080"
                symbol = "circle"
                size = 6
            
            # Enhanced description
            description_parts = []
            if entity.get('pelagios_data', {}).get('peripleo_description'):
                description_parts.append(entity['pelagios_data']['peripleo_description'][:100] + "...")
            if entity.get('pelagios_data', {}).get('temporal_bounds'):
                description_parts.append(f"Period: {entity['pelagios_data']['temporal_bounds']}")
            
            map_data.append({
                'Entity': entity['text'],
                'Type': entity['type'],
                'Latitude': entity['latitude'],
                'Longitude': entity['longitude'],
                'Source': source,
                'Color': color,
                'Symbol': symbol,
                'Size': size,
                'Description': ' | '.join(description_parts) if description_parts else entity.get('location_name', 'No description'),
                'Pelagios': '✅' if entity.get('pelagios_data') or entity.get('pleiades_id') else '❌'
            })
        
        df_map = pd.DataFrame(map_data)
        
        # Create enhanced Plotly map
        fig = px.scatter_mapbox(
            df_map,
            lat="Latitude",
            lon="Longitude",
            hover_name="Entity",
            hover_data={
                "Type": True,
                "Source": True,
                "Pelagios": True,
                "Description": True,
                "Latitude": ":.4f",
                "Longitude": ":.4f"
            },
            color="Source",
            color_discrete_map={
                "Pleiades": "#8B4513",
                "Peripleo": "#D2691E", 
                "Geopy": "#4169E1",
                "OpenStreetMap": "#228B22",
                "Other": "#808080"
            },
            size_max=15,
            zoom=2,
            height=600,
            title="Geographic Distribution of Historical and Modern Places"
        )
        
        fig.update_layout(
            mapbox_style="open-street-map",
            margin={"r": 0, "t": 40, "l": 0, "b": 0},
            showlegend=True
        )
        
        st.plotly_chart(fig, use_container_width=True)

    def render_enhanced_entity_table(self, entities: List[Dict[str, Any]]):
        """Enhanced entity table with better formatting."""
        if not entities:
            st.info("No entities found.")
            return
        
        # Prepare enhanced data for table
        table_data = []
        for entity in entities:
            row = {
                'Entity': entity['text'],
                'Type': entity['type'],
                'Pelagios': '✅' if entity.get('pelagios_data') else '',
                'Pleiades': '✅' if entity.get('pleiades_id') else '',
                'Links': self.format_enhanced_entity_links(entity),
                'Coordinates': f"{entity['latitude']:.4f}, {entity['longitude']:.4f}" if entity.get('latitude') else '',
                'Source': entity.get('geocoding_source', '')
            }
            
            # Enhanced description from multiple sources
            descriptions = []
            if entity.get('pelagios_data', {}).get('peripleo_description'):
                descriptions.append(f"Pelagios: {entity['pelagios_data']['peripleo_description'][:100]}...")
            if entity.get('wikidata_description'):
                descriptions.append(f"Wikidata: {entity['wikidata_description'][:100]}...")
            if entity.get('wikipedia_description'):
                descriptions.append(f"Wikipedia: {entity['wikipedia_description'][:100]}...")
            
            row['Description'] = ' | '.join(descriptions) if descriptions else entity.get('location_name', '')
            
            # Add temporal information
            if entity.get('pelagios_data', {}).get('temporal_bounds'):
                row['Period'] = entity['pelagios_data']['temporal_bounds']
            
            table_data.append(row)
        
        # Create enhanced DataFrame
        df = pd.DataFrame(table_data)
        
        # Display with enhanced formatting
        st.dataframe(
            df,
            use_container_width=True,
            column_config={
                "Entity": st.column_config.TextColumn("Entity", width="medium"),
                "Type": st.column_config.TextColumn("Type", width="small"),
                "Pelagios": st.column_config.TextColumn("", width="small"),
                "Pleiades": st.column_config.TextColumn("", width="small"),
                "Links": st.column_config.TextColumn("Links", width="medium"),
                "Coordinates": st.column_config.TextColumn("Coordinates", width="medium"),
                "Source": st.column_config.TextColumn("Geo Source", width="small"),
                "Description": st.column_config.TextColumn("Description", width="large"),
                "Period": st.column_config.TextColumn("Period", width="medium")
            }
        )

    def format_enhanced_entity_links(self, entity: Dict[str, Any]) -> str:
        """Enhanced link formatting with icons."""
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
        return " | ".join(links) if links else "No links"

    def render_enhanced_export_section(self, entities: List[Dict[str, Any]]):
        """Enhanced export section with better organization."""
        st.subheader("Pelagios Integration Exports")
        st.markdown("*Specialized formats for digital humanities and historical research*")
        
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
                help="Import this file into Recogito for collaborative annotation of historical texts",
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
                help="TEI XML with place name markup for digital humanities projects",
                use_container_width=True
            )
        
        with col3:
            # Enhanced JSON-LD export
            json_data = self.create_enhanced_jsonld_export(entities)
            json_str = json.dumps(json_data, indent=2, ensure_ascii=False)
            
            st.download_button(
                label="Download JSON-LD",
                data=json_str,
                file_name=f"{st.session_state.analysis_title}_entities.jsonld",
                mime="application/ld+json",
                help="Structured linked data export with Pelagios and Pleiades URIs",
                use_container_width=True
            )
        
        # Standard exports
        st.subheader("Standard Exports")
        st.markdown("*General-purpose formats for analysis and sharing*")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            # Enhanced HTML export
            if st.session_state.html_content:
                html_template = self.create_enhanced_html_export(entities)
                
                st.download_button(
                    label="Download HTML",
                    data=html_template,
                    file_name=f"{st.session_state.analysis_title}_entities.html",
                    mime="text/html",
                    help="Standalone HTML file with highlighted entities and metadata",
                    use_container_width=True
                )
        
        with col2:
            # Enhanced CSV export
            csv_data = self.create_enhanced_csv_export(entities)
            
            st.download_button(
                label="Download CSV",
                data=csv_data,
                file_name=f"{st.session_state.analysis_title}_entities.csv",
                mime="text/csv",
                help="Comprehensive CSV file for analysis in spreadsheet applications",
                use_container_width=True
            )
        
        with col3:
            # Summary report
            summary_report = self.create_summary_report(entities)
            
            st.download_button(
                label="Download Summary",
                data=summary_report,
                file_name=f"{st.session_state.analysis_title}_summary.txt",
                mime="text/plain",
                help="Human-readable summary of the analysis results",
                use_container_width=True
            )

    def create_enhanced_jsonld_export(self, entities: List[Dict[str, Any]]) -> Dict:
        """Enhanced JSON-LD export with comprehensive metadata."""
        json_data = {
            "@context": {
                "@vocab": "http://schema.org/",
                "pelagios": "http://pelagios.org/",
                "pleiades": "https://pleiades.stoa.org/places/",
                "crm": "http://www.cidoc-crm.org/cidoc-crm/",
                "geo": "http://www.w3.org/2003/01/geo/wgs84_pos#"
            },
            "@type": "TextDigitalDocument",
            "text": st.session_state.processed_text,
            "dateCreated": str(pd.Timestamp.now().isoformat()),
            "title": st.session_state.analysis_title,
            "generator": {
                "@type": "SoftwareApplication",
                "name": "EntityLinker with Pelagios Integration",
                "version": "2.1",
                "url": "https://github.com/your-repo/entity-linker"
            },
            "entities": []
        }
        
        # Enhanced entity formatting
        for entity in entities:
            entity_data = {
                "@type": self._get_schema_type(entity['type']),
                "name": entity['text'],
                "entityType": entity['type'],
                "startOffset": entity['start'],
                "endOffset": entity['end'],
                "confidence": entity.get('confidence', 1.0)
            }
            
            # Enhanced Pelagios data
            if entity.get('pelagios_data'):
                pelagios_data = entity['pelagios_data']
                entity_data['pelagios'] = {
                    "@type": "pelagios:Place",
                    "peripleo_id": pelagios_data.get('peripleo_id'),
                    "peripleo_url": pelagios_data.get('peripleo_url'),
                    "title": pelagios_data.get('peripleo_title'),
                    "description": pelagios_data.get('peripleo_description'),
                    "temporal_bounds": pelagios_data.get('temporal_bounds'),
                    "place_types": pelagios_data.get('place_types', [])
                }
            
            # Enhanced linking
            same_as = []
            if entity.get('pleiades_url'):
                same_as.append({
                    "@id": entity['pleiades_url'],
                    "@type": "pleiades:Place",
                    "source": "Pleiades"
                })
            if entity.get('wikidata_url'):
                same_as.append({
                    "@id": entity['wikidata_url'],
                    "source": "Wikidata"
                })
            if entity.get('wikipedia_url'):
                same_as.append({
                    "@id": entity['wikipedia_url'],
                    "source": "Wikipedia"
                })
            
            if same_as:
                entity_data['sameAs'] = same_as
            
            # Enhanced descriptions
            descriptions = {}
            if entity.get('pelagios_data', {}).get('peripleo_description'):
                descriptions['pelagios'] = entity['pelagios_data']['peripleo_description']
            if entity.get('wikidata_description'):
                descriptions['wikidata'] = entity['wikidata_description']
            if entity.get('wikipedia_description'):
                descriptions['wikipedia'] = entity['wikipedia_description']
            
            if descriptions:
                entity_data['descriptions'] = descriptions
            
            # Enhanced coordinates
            if entity.get('latitude') and entity.get('longitude'):
                entity_data['geo'] = {
                    "@type": "GeoCoordinates",
                    "latitude": entity['latitude'],
                    "longitude": entity['longitude'],
                    "source": entity.get('geocoding_source', 'unknown')
                }
                if entity.get('location_name'):
                    entity_data['geo']['name'] = entity['location_name']
            
            json_data['entities'].append(entity_data)
        
        return json_data

    def _get_schema_type(self, entity_type: str) -> str:
        """Map entity types to Schema.org types."""
        mapping = {
            'PERSON': 'Person',
            'ORGANIZATION': 'Organization',
            'GPE': 'Place',
            'LOCATION': 'Place',
            'FACILITY': 'Place',
            'NORP': 'Place',
            'ADDRESS': 'PostalAddress'
        }
        return mapping.get(entity_type, 'Thing')

    def create_enhanced_html_export(self, entities: List[Dict[str, Any]]) -> str:
        """Enhanced standalone HTML export with metadata."""
        # Statistics
        total_entities = len(entities)
        pelagios_enhanced = len([e for e in entities if e.get('pelagios_data') or e.get('pleiades_id')])
        geocoded = len([e for e in entities if e.get('latitude')])
        linked = len([e for e in entities if any([e.get('wikidata_url'), e.get('wikipedia_url'), e.get('britannica_url')])])
        
        html_template = f"""
        <!DOCTYPE html>
        <html lang="en">
        <head>
            <title>Entity Analysis: {st.session_state.analysis_title}</title>
            <meta charset="utf-8">
            <meta name="viewport" content="width=device-width, initial-scale=1">
            <meta name="generator" content="EntityLinker with Pelagios Integration v2.1">
            <style>
                body {{ 
                    font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif; 
                    max-width: 900px; 
                    margin: 0 auto; 
                    padding: 20px; 
                    background-color: #F5F0DC;
                }}
                .header {{ 
                    background: linear-gradient(135deg, #C4C3A2, #EFCA89); 
                    padding: 20px; 
                    border-radius: 10px; 
                    margin-bottom: 20px; 
                    box-shadow: 0 2px 10px rgba(0,0,0,0.1);
                }}
                .content {{ 
                    background: white; 
                    padding: 25px; 
                    border: 1px solid #E0D7C0; 
                    border-radius: 10px; 
                    line-height: 1.8; 
                    box-shadow: 0 2px 10px rgba(0,0,0,0.1);
                }}
                .statistics {{ 
                    background: #f9f9f9; 
                    padding: 20px; 
                    border-radius: 10px; 
                    margin-bottom: 20px; 
                    display: grid; 
                    grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); 
                    gap: 15px;
                }}
                .stat-item {{
                    background: white;
                    padding: 15px;
                    border-radius: 5px;
                    text-align: center;
                    border: 1px solid #E0D7C0;
                }}
                .stat-number {{
                    font-size: 2em;
                    font-weight: bold;
                    color: #C4A998;
                }}
                .legend {{
                    background: #f0f8ff;
                    padding: 15px;
                    border-radius: 5px;
                    margin-bottom: 20px;
                    border-left: 4px solid #C4C3A2;
                }}
                .pelagios-entity {{ border: 2px solid #D2691E !important; }}
                .pleiades-entity {{ border: 2px solid #8B4513 !important; }}
                @media (max-width: 768px) {{
                    body {{ padding: 10px; }}
                    .content, .header {{ padding: 15px; }}
                    .statistics {{ grid-template-columns: 1fr 1fr; }}
                }}
            </style>
        </head>
        <body>
            <div class="header">
                <h1>Entity Analysis: {st.session_state.analysis_title}</h1>
                <p>Generated on {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')} using EntityLinker v2.1 with Pelagios Integration</p>
            </div>
            
            <div class="statistics">
                <div class="stat-item">
                    <div class="stat-number">{total_entities}</div>
                    <div>Total Entities</div>
                </div>
                <div class="stat-item">
                    <div class="stat-number">{pelagios_enhanced}</div>
                    <div>Pelagios Enhanced</div>
                </div>
                <div class="stat-item">
                    <div class="stat-number">{geocoded}</div>
                    <div>Geocoded Places</div>
                </div>
                <div class="stat-item">
                    <div class="stat-number">{linked}</div>
                    <div>Linked Entities</div>
                </div>
            </div>
            
            <div class="legend">
                <h3>Visual Legend</h3>
                <p><strong>Pleiades entities</strong> (brown border) - Ancient world gazetteer</p>
                <p><strong>Peripleo entities</strong> (orange border) - Historical geography</p>
                <p><strong>Other linked entities</strong> (colored background) - General knowledge bases</p>
            </div>
            
            <div class="content">
                {st.session_state.html_content}
            </div>
            
            <div style="margin-top: 30px; padding: 20px; background: #f0f0f0; border-radius: 5px; text-align: center; font-size: 0.9em; color: #666;">
                Generated by <strong>EntityLinker with Pelagios Integration v2.1</strong><br>
                Enhanced historical place linking via Pelagios network services
            </div>
        </body>
        </html>
        """
        
        return html_template

    def create_enhanced_csv_export(self, entities: List[Dict[str, Any]]) -> str:
        """Enhanced CSV export with comprehensive data."""
        import io
        
        # Prepare comprehensive data for CSV
        csv_data = []
        for entity in entities:
            row = {
                'Entity': entity['text'],
                'Type': entity['type'],
                'Start_Position': entity['start'],
                'End_Position': entity['end'],
                'Confidence': entity.get('confidence', 1.0),
                
                # Pelagios data
                'Pelagios_Enhanced': 'Yes' if entity.get('pelagios_data') else 'No',
                'Pelagios_Title': entity.get('pelagios_data', {}).get('peripleo_title', ''),
                'Pelagios_Description': entity.get('pelagios_data', {}).get('peripleo_description', ''),
                'Temporal_Bounds': entity.get('pelagios_data', {}).get('temporal_bounds', ''),
                'Place_Types': ', '.join(entity.get('pelagios_data', {}).get('place_types', [])),
                
                # Pleiades data
                'Pleiades_ID': entity.get('pleiades_id', ''),
                'Pleiades_URL': entity.get('pleiades_url', ''),
                
                # Coordinates
                'Has_Coordinates': 'Yes' if entity.get('latitude') else 'No',
                'Latitude': entity.get('latitude', ''),
                'Longitude': entity.get('longitude', ''),
                'Geocoding_Source': entity.get('geocoding_source', ''),
                'Location_Name': entity.get('location_name', ''),
                'Search_Term_Used': entity.get('search_term_used', ''),
                
                # External links
                'Wikidata_URL': entity.get('wikidata_url', ''),
                'Wikipedia_URL': entity.get('wikipedia_url', ''),
                'Britannica_URL': entity.get('britannica_url', ''),
                
                # Descriptions
                'Wikidata_Description': entity.get('wikidata_description', ''),
                'Wikipedia_Description': entity.get('wikipedia_description', ''),
                'Wikipedia_Title': entity.get('wikipedia_title', ''),
                'Britannica_Title': entity.get('britannica_title', ''),
                
                # Peripleo URLs
                'Peripleo_URL': entity.get('pelagios_data', {}).get('peripleo_url', '')
            }
            csv_data.append(row)
        
        # Convert to CSV
        df = pd.DataFrame(csv_data)
        return df.to_csv(index=False)

    def create_summary_report(self, entities: List[Dict[str, Any]]) -> str:
        """Create a human-readable summary report."""
        total_entities = len(entities)
        pelagios_enhanced = len([e for e in entities if e.get('pelagios_data')])
        pleiades_linked = len([e for e in entities if e.get('pleiades_id')])
        geocoded = len([e for e in entities if e.get('latitude')])
        
        # Entity type breakdown
        type_counts = {}
        for entity in entities:
            entity_type = entity['type']
            type_counts[entity_type] = type_counts.get(entity_type, 0) + 1
        
        # Pelagios entities
        pelagios_entities = [e for e in entities if e.get('pelagios_data') or e.get('pleiades_id')]
        
        report = f"""
ENTITY LINKING ANALYSIS SUMMARY
===============================

Analysis Title: {st.session_state.analysis_title}
Generated: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}
Text Length: {len(st.session_state.processed_text)} characters

OVERVIEW
--------
Total Entities Found: {total_entities}
Pelagios Enhanced: {pelagios_enhanced}
Pleiades Linked: {pleiades_linked}
Geocoded Places: {geocoded}
Success Rate: {(pelagios_enhanced + pleiades_linked) / total_entities * 100:.1f}% for historical places

ENTITY TYPES
------------
"""
        
        for entity_type, count in sorted(type_counts.items()):
            percentage = count / total_entities * 100
            report += f"{entity_type}: {count} ({percentage:.1f}%)\n"
        
        if pelagios_entities:
            report += f"""

PELAGIOS & PLEIADES ENHANCED PLACES
==================================
"""
            for entity in pelagios_entities[:10]:  # Top 10
                report += f"\n{entity['text']} ({entity['type']})\n"
                if entity.get('pleiades_id'):
                    report += f"  Pleiades ID: {entity['pleiades_id']}\n"
                if entity.get('pelagios_data', {}).get('temporal_bounds'):
                    report += f"  Period: {entity['pelagios_data']['temporal_bounds']}\n"
                if entity.get('latitude'):
                    report += f"  Coordinates: {entity['latitude']:.4f}, {entity['longitude']:.4f}\n"
                if entity.get('pelagios_data', {}).get('peripleo_description'):
                    desc = entity['pelagios_data']['peripleo_description'][:100] + "..."
                    report += f"  Description: {desc}\n"
        
        report += f"""

SOURCES USED
============
- Pelagios Peripleo: Historical geography search
- Pleiades Gazetteer: Ancient world places (direct API)
- Wikidata: General knowledge graph
- Wikipedia: Encyclopedia articles
- Britannica: Academic encyclopedia
- OpenStreetMap/Nominatim: Modern geocoding
- Geopy: Python geocoding libraries

METHODOLOGY
===========
1. spaCy Named Entity Recognition with enhanced validation
2. Pelagios network integration for historical places
3. Multi-source entity linking with fallback strategies
4. Contextual geocoding with geographical awareness
5. Export to standard formats (TEI XML, Recogito JSON, JSON-LD)

This analysis was generated using EntityLinker v2.1 with Pelagios Integration.
For more information, visit: https://github.com/your-repo/entity-linker
"""
        
        return report

    def run(self):
        """Enhanced main application runner."""
        # Enhanced CSS styling
        st.markdown("""
        <style>
        .stApp {
            background: linear-gradient(135deg, #F5F0DC 0%, #F8F5E4 100%) !important;
        }
        .main .block-container {
            background-color: transparent !important;
            padding-top: 2rem;
        }
        .stSidebar {
            background: linear-gradient(180deg, #F5F0DC 0%, #E8E1D4 100%) !important;
        }
        
        /* Enhanced form styling */
        .stSelectbox > div > div, .stTextInput > div > div > input, .stTextArea > div > div > textarea {
            background-color: white !important;
            border: 1px solid #E0D7C0 !important;
            border-radius: 8px !important;
        }
        
        /* Enhanced expander styling */
        .stExpander {
            background-color: white !important;
            border: 1px solid #E0D7C0 !important;
            border-radius: 8px !important;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1) !important;
        }
        
        /* Enhanced dataframe styling */
        .stDataFrame {
            background-color: white !important;
            border-radius: 8px !important;
            overflow: hidden !important;
        }
        
        /* Enhanced button styling */
        .stButton > button {
            background: linear-gradient(135deg, #C4A998 0%, #B5998A 100%) !important;
            color: black !important;
            border: none !important;
            border-radius: 8px !important;
            font-weight: 600 !important;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1) !important;
            transition: all 0.3s ease !important;
        }
        .stButton > button:hover {
            background: linear-gradient(135deg, #B5998A 0%, #A68977 100%) !important;
            transform: translateY(-1px) !important;
            box-shadow: 0 4px 8px rgba(0,0,0,0.15) !important;
        }
        
        /* Enhanced metric styling */
        .metric-container {
            background: white;
            padding: 1rem;
            border-radius: 8px;
            border: 1px solid #E0D7C0;
            box-shadow: 0 2px 4px rgba(0,0,0,0.05);
        }
        </style>
        """, unsafe_allow_html=True)
        
        # Render enhanced header
        self.render_header()
        
        # Render enhanced sidebar
        self.render_sidebar()
        
        # Enhanced input section
        text_input, analysis_title = self.render_input_section()
        
        # Enhanced process button
        if st.button("🚀 Process Text with Enhanced Pelagios Integration", type="primary", use_container_width=True):
            if text_input.strip():
                self.process_text(text_input, analysis_title)
            else:
                st.warning("⚠️ Please enter some text to analyze.")
        
        # Add separator
        st.markdown("---")
        
        # Enhanced results section
        self.render_results()
        
        # Footer
        st.markdown("""
        <div style="margin-top: 50px; padding: 20px; background: white; border-radius: 10px; border: 1px solid #E0D7C0; text-align: center;">
            <p style="color: #666; margin: 0;"><strong>EntityLinker v2.1</strong> with Enhanced Pelagios Integration</p>
            <p style="color: #666; margin: 5px 0 0 0; font-size: 0.9em;">
                Linking modern NLP with historical geography through the Pelagios network
            </p>
        </div>
        """, unsafe_allow_html=True)


def main():
    """Main function to run the enhanced Streamlit application."""
    app = StreamlitEntityLinker()
    app.run()


if __name__ == "__main__":
    main()                    entity['context_used'] = str(context_info)
                    return True
                
                time.sleep(0.5)  # Rate limiting
            except (GeocoderTimedOut, GeocoderServiceError):
                continue
            except Exception:
                continue
        
        return False
    
    def _try_ollama_contextual_geocoding(self, entity, context_info, search_terms):
        """
        Use Ollama for intelligent place disambiguation based on context.
        """
        import requests
        import json
        from geopy.geocoders import Nominatim
        
        try:
            # Create context-aware prompt
            prompt = f"""Given the historical context and place name, suggest the most likely modern location for geocoding.

Place name: {entity['text']}
Historical context:
- Regions: {', '.join(context_info.get('regions', []))}
- Time period: {context_info.get('time_period', 'Unknown')}
- Civilizations: {', '.join(context_info.get('ancient_civilizations', []))}

Consider that {entity['text']} might be:
1. An ancient city that still exists today
2. An archaeological site near a modern city
3. A historical name for a modern location

Provide ONE specific geocoding search string that would find this place on a modern map.
Examples: "Thebes, Greece", "Alexandria, Egypt", "Troy archaeological site, Turkey"

Response:"""

            response = requests.post(
                'http://localhost:11434/api/generate',
                json={
                    'model': 'mistral',
                    'prompt': prompt,
                    'stream': False,
                    'options': {'temperature': 0.3}
                },
                timeout=15
            )
            
            if response.status_code == 200:
                result = response.json()
                suggested_query = result.get('response', '').strip()
                
                # Clean up the response
                suggested_query = suggested_query.replace('"', '').replace("'", '').strip()
                
                if suggested_query and len(suggested_query) < 100:
                    # Try geocoding with the suggested query
                    geolocator = Nominatim(user_agent="EntityLinker-Ollama-Context", timeout=10)
                    location = geolocator.geocode(suggested_query)
                    
                    if location:
                        entity['latitude'] = location.latitude
                        entity['longitude'] = location.longitude
                        entity['location_name'] = location.address
                        entity['geocoding_source'] = 'geopy_ollama_contextual'
                        entity['search_term_used'] = suggested_query
                        entity['context_used'] = str(context_info)
                        return True
                        
        except Exception as e:
            print(f"Ollama contextual geocoding failed: {e}")
        
        return False
    
    def _try_python_geocoding(self, entity):
        """Enhanced Python geocoding with multiple services."""
        try:
            from geopy.geocoders import Nominatim, ArcGIS
            from geopy.exc import GeocoderTimedOut, GeocoderServiceError
            
            geocoders = [
                ('nominatim', Nominatim(user_agent="EntityLinker/2.1", timeout=15)),
                ('arcgis', ArcGIS(timeout=15)),
            ]
            
            for name, geocoder in geocoders:
                try:
                    location = geocoder.geocode(entity['text'], timeout=15)
                    if location:
                        entity['latitude'] = location.latitude
                        entity['longitude'] = location.longitude
                        entity['location_name'] = location.address
                        entity['geocoding_source'] = f'geopy_{name}'
                        return True
                        
                    time.sleep(0.4)
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
        """Enhanced OpenStreetMap geocoding with better error handling."""
        try:
            url = "https://nominatim.openstreetmap.org/search"
            params = {
                'q': entity['text'],
                'format': 'json',
                'limit': 1,
                'addressdetails': 1,
                'accept-language': 'en'
            }
            headers = {
                'User-Agent': 'EntityLinker/2.1 (Academic Research)',
                'Accept': 'application/json'
            }
        
            response = requests.get(url, params=params, headers=headers, timeout=15)
            if response.status_code == 200:
                data = response.json()
                if data and len(data) > 0:
                    result = data[0]
                    entity['latitude'] = float(result['lat'])
                    entity['longitude'] = float(result['lon'])
                    entity['location_name'] = result['display_name']
                    entity['geocoding_source'] = 'openstreetmap'
                    return True
        
            time.sleep(0.4)  # Rate limiting
        except Exception as e:
            print(f"OpenStreetMap geocoding failed for {entity['text']}: {e}")
            pass
        
        return False
    
    def _try_aggressive_geocoding_with_context(self, entity, context_info):
        """
        Enhanced aggressive geocoding that uses both Ollama and context for better results.
        """
        import requests
        import time
        
        # First try Ollama if available
        try:
            response = requests.get('http://localhost:11434/api/tags', timeout=1)
            if response.status_code == 200:
                # Create highly context-aware prompt
                prompt = f"""The place "{entity['text']}" appears in a text about:
- Regions: {', '.join(context_info.get('regions', ['unknown']))}
- Time period: {context_info.get('time_period', 'unknown')}
- Civilizations: {', '.join(context_info.get('ancient_civilizations', ['unknown']))}

This place might be ancient, historical, or mythological. Suggest the most likely modern location or archaeological site that corresponds to "{entity['text']}".

Consider historical name changes, ancient sites near modern cities, and regional variations.

Provide ONE specific search query for modern geocoding. Examples:
- "Troy archaeological site, Turkey"
- "Thebes, Greece" (not Thebes, Egypt)
- "Ancient Babylon site, Iraq"

Response:"""

                response = requests.post(
                    'http://localhost:11434/api/generate',
                    json={
                        'model': 'mistral',
                        'prompt': prompt,
                        'stream': False,
                        'options': {'temperature': 0.3}
                    },
                    timeout=20
                )
                
                if response.status_code == 200:
                    result = response.json()
                    search_term = result.get('response', '').strip().replace('"', '').replace("'", '')
                    
                    if search_term and len(search_term) < 100:
                        # Try geocoding
                        url = "https://nominatim.openstreetmap.org/search"
                        params = {
                            'q': search_term,
                            'format': 'json',
                            'limit': 1,
                            'addressdetails': 1
                        }
                        
                        geo_response = requests.get(url, params=params, timeout=10)
                        if geo_response.status_code == 200:
                            data = geo_response.json()
                            if data:
                                result = data[0]
                                entity['latitude'] = float(result['lat'])
                                entity['longitude'] = float(result['lon'])
                                entity['location_name'] = result['display_name']
                                entity['geocoding_source'] = 'openstreetmap_ollama_aggressive'
                                entity['search_term_used'] = search_term
                                entity['context_used'] = str(context_info)
                                return True
                        
                        time.sleep(0.4)
        except Exception as e:
            print(f"Ollama aggressive geocoding failed: {e}")
        
        # Fallback: Build context-aware variations
        fallback_variations = []
        
        # Add region-specific variations
        if context_info.get('regions'):
            for region in context_info['regions'][:2]:
                if region == 'Greece':
                    fallback_variations.extend([
                        f"{entity['text']}, Greece",
                        f"ancient {entity['text']}, Greece"
                    ])
                elif region == 'Egypt':
                    fallback_variations.extend([
                        f"{entity['text']}, Egypt",
                        f"ancient {entity['text']}, Egypt"
                    ])
                elif region == 'Near East':
                    fallback_variations.extend([
                        f"{entity['text']}, Turkey",
                        f"{entity['text']}, Syria",
                        f"{entity['text']}, Iraq"
                    ])
                elif region == 'Persia':
                    fallback_variations.extend([
                        f"{entity['text']}, Iran",
                        f"ancient {entity['text']}, Iran"
                    ])
        
        # Add time-period specific variations
        if context_info.get('time_period') and 'BCE' in str(context_info.get('time_period', '')):
            fallback_variations.extend([
                f"ancient {entity['text']}",
                f"{entity['text']} archaeological site",
                f"{entity['text']} ruins"
            ])
        
        # Add civilization-specific variations
        if context_info.get('ancient_civilizations'):
            for civ in context_info['ancient_civilizations']:
                if 'Greek' in civ or 'Hellenistic' in civ:
                    fallback_variations.append(f"{entity['text']}, ancient Greece")
                elif 'Roman' in civ:
                    fallback_variations.append(f"{entity['text']}, Roman site")
                elif 'Egyptian' in civ:
                    fallback_variations.append(f"{entity['text']}, ancient Egypt")
        
        # Remove duplicates while preserving order
        seen = set()
        unique_variations = []
        for v in fallback_variations:
            if v not in seen:
                seen.add(v)
                unique_variations.append(v)
        
        # Try each variation
        for term in unique_variations[:5]:  # Limit to top 5 variations
            try:
                url = "https://nominatim.openstreetmap.org/search"
                params = {
                    'q': term,
                    'format': 'json',
                    'limit': 1,
                    'addressdetails': 1,
                    'accept-language': 'en'
                }
                headers = {
                    'User-Agent': 'EntityLinker/2.1 (Academic Research)',
                    'Accept': 'application/json'
                }
                
                response = requests.get(url, params=params, headers=headers, timeout=15)
                if response.status_code == 200:
                    data = response.json()
                    if data:
                        result = data[0]
                        entity['latitude'] = float(result['lat'])
                        entity['longitude'] = float(result['lon'])
                        entity['location_name'] = result['display_name']
                        entity['geocoding_source'] = 'openstreetmap_contextual_fallback'
                        entity['search_term_used'] = term
                        entity['context_used'] = str(context_info)
                        return True
                time.sleep(0.3)
            except Exception:
                continue
        
        return False

    def link_to_wikidata(self, entities):
        """Enhanced Wikidata linking with better search strategies."""
        import requests
        import time
        
        for entity in entities:
            try:
                # Try multiple search strategies
                search_terms = [
                    entity['text'],
                    f"{entity['text']} ancient",
                    f"{entity['text']} city" if entity['type'] in ['GPE', 'LOCATION'] else entity['text']
                ]
                
                for search_term in search_terms:
                    url = "https://www.wikidata.org/w/api.php"
                    params = {
                        'action': 'wbsearchentities',
                        'format': 'json',
                        'search': search_term,
                        'language': 'en',
                        'limit': 3,
                        'type': 'item'
                    }
                    
                    response = requests.get(url, params=params, timeout=10)
                    if response.status_code == 200:
                        data = response.json()
                        if data.get('search') and len(data['search']) > 0:
                            # Look for best match
                            for result in data['search']:
                                # Prefer exact matches or close matches
                                if (result['label'].lower() == entity['text'].lower() or
                                    entity['text'].lower() in result['label'].lower()):
                                    entity['wikidata_url'] = f"http://www.wikidata.org/entity/{result['id']}"
                                    entity['wikidata_description'] = result.get('description', '')
                                    break
                            else:
                                # If no perfect match, use first result
                                result = data['search'][0]
                                entity['wikidata_url'] = f"http://www.wikidata.org/entity/{result['id']}"
                                entity['wikidata_description'] = result.get('description', '')
                            break
                
                time.sleep(0.2)  # Rate limiting
            except Exception:
                pass  # Continue if API call fails
        
        return entities

    def link_to_wikipedia(self, entities):
        """Enhanced Wikipedia linking with better search and disambiguation."""
        import requests
        import time
        import urllib.parse
        
        for entity in entities:
            # Skip if already has Wikidata link
            if entity.get('wikidata_url'):
                continue
                
            try:
                # Try multiple search strategies
                search_terms = [
                    entity['text'],
                    f"{entity['text']} ancient",
                    f"{entity['text']} city" if entity['type'] in ['GPE', 'LOCATION'] else entity['text'],
                    f"{entity['text']} (ancient city)" if entity['type'] in ['GPE', 'LOCATION'] else entity['text']
                ]
                
                for search_term in search_terms:
                    # Use Wikipedia's search API
                    search_url = "https://en.wikipedia.org/w/api.php"
                    search_params = {
                        'action': 'query',
                        'format': 'json',
                        'list': 'search',
                        'srsearch': search_term,
                        'srlimit': 3,
                        'srprop': 'snippet|titlesnippet'
                    }
                    
                    headers = {'User-Agent': 'EntityLinker/2.1 (Academic Research)'}
                    response = requests.get(search_url, params=search_params, headers=headers, timeout=15)
                    
                    if response.status_code == 200:
                        data = response.json()
                        if data.get('query', {}).get('search'):
                            # Look for best match
                            for result in data['query']['search']:
                                page_title = result['title']
                                
                                # Prefer exact matches or close matches
                                if (entity['text'].lower() in page_title.lower() or
                                    any(word in page_title.lower() for word in entity['text'].lower().split())):
                                    
                                    # Create Wikipedia URL
                                    encoded_title = urllib.parse.quote(page_title.replace(' ', '_'))
                                    entity['wikipedia_url'] = f"https://en.wikipedia.org/wiki/{encoded_title}"
                                    entity['wikipedia_title'] = page_title
                                    
                                    # Get a snippet/description from the search result
                                    if result.get('snippet'):
                                        import re
                                        snippet = re.sub(r'<[^>]+>', '', result['snippet'])
                                        entity['wikipedia_description'] = snippet[:200] + "..." if len(snippet) > 200 else snippet
                                    break
                            
                            if entity.get('wikipedia_url'):
                                break
                
                time.sleep(0.3)  # Rate limiting
            except Exception as e:
                print(f"Wikipedia linking failed for {entity['text']}: {e}")
                pass
        
        return entities

    def link_to_britannica(self, entities):
        """Enhanced Britannica linking with better search patterns.""" 
        import requests
        import re
        import time
        
        for entity in entities:
            # Skip if already has Wikidata or Wikipedia link
            if entity.get('wikidata_url') or entity.get('wikipedia_url'):
                continue
                
            try:
                search_terms = [
                    entity['text'],
                    f"{entity['text']} ancient",
                    f"{entity['text']} city" if entity['type'] in ['GPE', 'LOCATION'] else entity['text']
                ]
                
                for search_term in search_terms:
                    search_url = "https://www.britannica.com/search"
                    params = {'query': search_term}
                    headers = {
                        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
                    }
                    
                    response = requests.get(search_url, params=params, headers=headers, timeout=15)
                    if response.status_code == 200:
                        # Look for article links with improved patterns
                        patterns = [
                            r'href="(/topic/[^"]*)"[^>]*>([^<]*)</a>',
                            r'href="(/place/[^"]*)"[^>]*>([^<]*)</a>',
                            r'href="(/biography/[^"]*)"[^>]*>([^<]*)</a>'
                        ]
                        
                        for pattern in patterns:
                            matches = re.findall(pattern, response.text)
                            for url_path, link_text in matches:
                                if (entity['text'].lower() in link_text.lower() or 
                                    link_text.lower() in entity['text'].lower() or
                                    any(word in link_text.lower() for word in entity['text'].lower().split())):
                                    entity['britannica_url'] = f"https://www.britannica.com{url_path}"
                                    entity['britannica_title'] = link_text.strip()
                                    break
                            
                            if entity.get('britannica_url'):
                                break
                    
                    if entity.get('britannica_url'):
                        break
                
                time.sleep(0.4)  # Rate limiting
            except Exception:
                pass
        
        return entities


class StreamlitEntityLinker:
    """
    Enhanced Streamlit wrapper for the EntityLinker class with improved Pelagios integration.
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
    def cached_enhance_with_pelagios(_self, entities_json: str, context_json: str) -> str:
        """Cached Pelagios enhancement with context."""
        import json
        entities = json.loads(entities_json)
        context_info = json.loads(context_json) if context_json else None
        enhanced_entities = _self.pelagios.enhance_entities_with_pelagios(entities, context_info)
        return json.dumps(enhanced_entities, default=str)

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
        st.markdown("**Extract and link named entities from text to external knowledge bases including historical gazetteers**")
        
        # Enhanced process diagram
        st.markdown("""
        <div style="background-color: white; padding: 20px; border-radius: 10px; margin: 20px 0; border: 1px solid #E0D7C0;">
            <div style="text-align: center; margin-bottom: 20px;">
                <div style="background-color: #C4C3A2; padding: 10px; border-radius: 5px; display: inline-block; margin: 5px;">
                     <strong>Input Text</strong>
                </div>
                <div style="margin: 10px 0;">⬇️</div>
                <div style="background-color: #9fd2cd; padding: 10px; border-radius: 5px; display: inline-block; margin: 5px;">
                     <strong>Enhanced spaCy Entity Recognition</strong>
                </div>
                <div style="margin: 10px 0;">⬇️</div>
                <div style="text-align: center;">
                    <strong>Multi-Source Entity Linking:</strong>
                </div>
                <div style="margin: 15px 0;">
                    <div style="background-color: #EFCA89; padding: 8px; border-radius: 5px; display: inline-block; margin: 3px; font-size: 0.9em;">
                         <strong>Pelagios/Peripleo</strong><br><small>Historical geography with fallbacks</small>
                    </div>
                    <div style="background-color: #C3B5AC; padding: 8px; border-radius: 5px; display: inline-block; margin: 3px; font-size: 0.9em;">
                        <strong>Pleiades Direct</strong><br><small>Ancient world gazetteer</small>
                    </div>
                    <div style="background-color: #BF7B69; padding: 8px; border-radius: 5px; display: inline-block; margin: 3px; font-size: 0.9em;">
                         <strong>Wikidata/Wikipedia</strong><br><small>General knowledge</small>
                    </div>
                    <div style="background-color: #D4C5B9; padding: 8px; border-radius: 5px; display: inline-block; margin: 3px; font-size: 0.9em;">
                         <strong>Enhanced Geocoding</strong><br><small>Multiple mapping services</small>
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
        """Render the sidebar with enhanced information."""
        # Entity linking information
        st.sidebar.subheader("Entity Linking")
        st.sidebar.info("Enhanced multi-source linking: Pelagios historical places, direct Pleiades search, Wikidata, Wikipedia, and Britannica as fallbacks. Places are geocoded using multiple services with contextual awareness.")
        
        # Pelagios information
        st.sidebar.subheader("Pelagios Integration")
        st.sidebar.info("Historical places are enhanced with data from Peripleo with direct Pleiades fallback. Includes temporal bounds, place types, and authoritative ancient world coordinates.")
        
        # Export formats
        st.sidebar.subheader("Export Formats")
        st.sidebar.info("Results exported as Recogito annotations for collaborative markup, TEI XML for digital humanities, or JSON-LD for linked data applications.")
        
        # Debugging information
        st.sidebar.subheader("Debug Information")
        if st.sidebar.checkbox("Show debug output"):
            st.session_state.show_debug = True
        else:
            st.session_state.show_debug = False

    def render_input_section(self):
        """Render the enhanced text input section."""
        st.header("Input Text")
        
        # Add title input
        analysis_title = st.text_input(
            "Analysis Title (optional)",
            placeholder="Enter a title for this analysis...",
            help="This will be used for naming output files"
        )
        
        # Enhanced sample text - Herodotus passage for better historical testing
        sample_text = """The Persian learned men say that the Phoenicians were the cause of the dispute. These (they say) came to our seas from the sea which is called Red, and having settled in the country which they still occupy, at once began to make long voyages. Among other places to which they carried Egyptian and Assyrian merchandise, they came to Argos, which was at that time preeminent in every way among the people of what is now called Hellas. The Phoenicians came to Argos, and set out their cargo. On the fifth or sixth day after their arrival, when their wares were almost all sold, many women came to the shore and among them especially the daughter of the king, whose name was Io (according to Persians and Greeks alike), the daughter of Inachus. As these stood about the stern of the ship bargaining for the wares they liked, the Phoenicians incited one another to set upon them. Most of the women escaped: Io and others were seized and thrown into the ship, which then sailed away for Egypt."""       
        
        # Text input area with sample
        text_input = st.text_area(
            "Enter your text here:",
            value=sample_text,
            height=200,
            placeholder="Paste your text here for entity extraction...",
            help="This sample from Herodotus includes ancient places perfect for testing Pelagios integration: Argos, Egypt, Assyria, Hellas, and more!"
        )
        
        # File upload option
        with st.expander("📁 Or upload a text file"):
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
        
        # Use suggested title if no title provided
        if not analysis_title and hasattr(st.session_state, 'suggested_title'):
            analysis_title = st.session_state.suggested_title
        elif not analysis_title and not uploaded_file:
            analysis_title = "herodotus_histories"
        
        return text_input, analysis_title or "text_analysis"

    def process_text(self, text: str, title: str):
        """
        Enhanced text processing with improved error handling and debugging.
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
                # Create enhanced progress tracking
                progress_bar = st.progress(0)
                status_text = st.empty()
                debug_container = st.expander("🔧 Processing Details", expanded=#!/usr/bin/env python3
"""
Streamlit Entity Linker Application with Fixed Pelagios Integration

A web interface for Named Entity Recognition using spaCy with enhanced
linking to Pelagios network services including Peripleo, Pleiades, and 
Recogito-compatible exports.

Author: EntityLinker Team
Version: 2.1 - Fixed Pelagios Integration
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
    """Completely Fixed Pelagios integration focusing on what actually works."""
    
    def __init__(self):
        """Initialize Pelagios integration with working endpoints."""
        # Use the working legacy API endpoints
        self.peripleo_legacy_url = "http://peripleo.pelagios.org/peripleo/search"
        self.pleiades_base_url = "https://pleiades.stoa.org"
        self.pleiades_search_url = "https://pleiades.stoa.org/search_rss"
        
        # Session for connection pooling
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'EntityLinker-Pelagios/2.1 (Academic Research)',
            'Accept': 'application/json, application/xml, text/xml, */*'
        })
        
    def search_peripleo_legacy(self, place_name: str, limit: int = 5) -> List[Dict]:
        """
        Search the legacy Peripleo API with proper parameters.
        """
        print(f"Searching Peripleo legacy API for: '{place_name}'")
        
        # Use the documented legacy API parameters
        search_params = {
            'query': place_name,
            'types': 'place',  # Only search for places
            'limit': limit,
            'prettyprint': 'true'  # For easier debugging
        }
        
        try:
            print(f"Making request to: {self.peripleo_legacy_url}")
            print(f"Parameters: {search_params}")
            
            response = self.session.get(
                self.peripleo_legacy_url,
                params=search_params,
                timeout=30  # Longer timeout for potentially slow API
            )
            
            print(f"Response status: {response.status_code}")
            print(f"Full URL: {response.url}")
            
            if response.status_code == 200:
                print(f"Success! Response length: {len(response.text)} chars")
                
                # Try to parse as JSON first
                try:
                    data = response.json()
                    print(f"JSON Response structure: {type(data)}")
                    
                    if isinstance(data, dict):
                        print(f"JSON keys: {list(data.keys())}")
                        
                        # Look for results in various possible keys
                        items = []
                        for key in ['items', 'results', 'places', 'features']:
                            if key in data and isinstance(data[key], list):
                                items = data[key]
                                print(f"Found {len(items)} items in '{key}' field")
                                break
                        
                        if items:
                            # Process the items
                            processed_items = []
                            for item in items[:limit]:
                                processed_item = self._process_peripleo_item(item)
                                if processed_item:
                                    processed_items.append(processed_item)
                                    print(f"Processed: {processed_item.get('title', 'Untitled')}")
                            
                            return processed_items
                        else:
                            print(f"No items found in response: {str(data)[:200]}...")
                    
                    elif isinstance(data, list):
                        print(f"Direct list response with {len(data)} items")
                        processed_items = []
                        for item in data[:limit]:
                            processed_item = self._process_peripleo_item(item)
                            if processed_item:
                                processed_items.append(processed_item)
                        return processed_items
                
                except json.JSONDecodeError:
                    print("Response is not JSON, trying as XML/HTML...")
                    print(f"Response preview: {response.text[:500]}...")
                    
                    # Check if it's an error page or maintenance message
                    if "maintenance" in response.text.lower() or "gateway" in response.text.lower():
                        print("API appears to be under maintenance")
                        return []
                    
                    # Try to parse as XML (unlikely but possible)
                    return self._try_parse_xml_response(response.text, place_name)
            
            else:
                print(f"HTTP Error {response.status_code}")
                print(f"Error response: {response.text[:300]}...")
                
        except requests.RequestException as e:
            print(f"Request failed: {e}")
        except Exception as e:
            print(f"Unexpected error: {e}")
        
        return []
    
    def _process_peripleo_item(self, item: Dict) -> Optional[Dict]:
        """Process a Peripleo API response item."""
        try:
            # Extract basic information
            processed = {
                'title': item.get('title', item.get('label', item.get('name', ''))),
                'identifier': item.get('identifier', item.get('uri', item.get('id', ''))),
                'description': item.get('description', ''),
                'object_type': item.get('object_type', 'place'),
                'source_gazetteer': 'peripleo'
            }
            
            # Extract temporal bounds
            if item.get('temporal_bounds'):
                tb = item['temporal_bounds']
                if isinstance(tb, dict):
                    start = tb.get('start', '')
                    end = tb.get('end', '')
                    if start or end:
                        processed['temporal_bounds'] = f"{start}-{end}" if start != end else str(start)
            
            # Extract geographical bounds
            if item.get('geo_bounds'):
                gb = item['geo_bounds']
                if isinstance(gb, dict) and all(k in gb for k in ['min_lat', 'max_lat', 'min_lon', 'max_lon']):
                    # Calculate centroid
                    lat = (gb['min_lat'] + gb['max_lat']) / 2
                    lon = (gb['min_lon'] + gb['max_lon']) / 2
                    processed['geo_bounds'] = {
                        'centroid': {'lat': lat, 'lon': lon}
                    }
            
            # Check if this is from Pleiades
            identifier = processed['identifier']
            if 'pleiades.stoa.org' in identifier:
                processed['source_gazetteer'] = 'pleiades'
                if '/places/' in identifier:
                    processed['source_id'] = identifier.split('/places/')[-1].rstrip('/')
            
            return processed
            
        except Exception as e:
            print(f"Error processing item {item}: {e}")
            return None
    
    def _try_parse_xml_response(self, xml_text: str, place_name: str) -> List[Dict]:
        """Try to parse XML response (for RSS feeds, etc.)."""
        try:
            root = ET.fromstring(xml_text)
            items = []
            
            # Look for RSS items
            for item in root.findall('.//item'):
                title_elem = item.find('title')
                link_elem = item.find('link')
                
                if title_elem is not None and link_elem is not None:
                    title = title_elem.text or ''
                    link = link_elem.text or ''
                    
                    if place_name.lower() in title.lower():
                        items.append({
                            'title': title,
                            'identifier': link,
                            'description': '',
                            'source_gazetteer': 'pleiades' if 'pleiades' in link else 'unknown'
                        })
            
            return items
            
        except ET.ParseError:
            return []
    
    def search_pleiades_directly(self, place_name: str) -> List[Dict]:
        """
        Direct search of Pleiades gazetteer - this usually works!
        """
        print(f"Searching Pleiades directly for: '{place_name}'")
        
        try:
            params = {
                'SearchableText': place_name,
                'portal_type': 'Place',
                'review_state': 'published',
                'sort_on': 'effective'
            }
            
            response = self.session.get(
                self.pleiades_search_url,
                params=params,
                timeout=20
            )
            
            print(f"Pleiades URL: {response.url}")
            print(f"Status: {response.status_code}")
            
            if response.status_code == 200:
                items = []
                
                try:
                    # Parse RSS/XML response
                    root = ET.fromstring(response.text)
                    print(f"Parsed XML successfully")
                    
                    for item in root.findall('.//item'):
                        title_elem = item.find('title')
                        link_elem = item.find('link')
                        desc_elem = item.find('description')
                        
                        if title_elem is not None and link_elem is not None:
                            title = title_elem.text.strip()
                            link = link_elem.text.strip()
                            description = desc_elem.text.strip() if desc_elem is not None else ''
                            
                            # Extract Pleiades ID
                            pleiades_id = None
                            if '/places/' in link:
                                pleiades_id = link.split('/places/')[-1].rstrip('/').split('?')[0]
                            
                            if pleiades_id and pleiades_id.isdigit():
                                print(f"Found Pleiades place: {title} (ID: {pleiades_id})")
                                
                                item_data = {
                                    'title': title,
                                    'identifier': link,
                                    'description': description,
                                    'source_gazetteer': 'pleiades',
                                    'source_id': pleiades_id,
                                    'pleiades_url': link
                                }
                                
                                # Try to get JSON details for coordinates
                                json_details = self._get_pleiades_json_details(pleiades_id)
                                if json_details:
                                    item_data.update(json_details)
                                
                                items.append(item_data)
                
                except ET.ParseError as e:
                    print(f"Error parsing Pleiades RSS: {e}")
                    print(f"Response preview: {response.text[:300]}...")
                
                print(f"Found {len(items)} Pleiades items")
                return items
            else:
                print(f"Pleiades search failed: {response.status_code}")
                
        except Exception as e:
            print(f"Pleiades search error: {e}")
        
        return []
    
    def _get_pleiades_json_details(self, pleiades_id: str) -> Optional[Dict]:
        """Get detailed information from Pleiades JSON API."""
        try:
            json_url = f"{self.pleiades_base_url}/places/{pleiades_id}/json"
            print(f"Fetching Pleiades JSON: {json_url}")
            
            response = self.session.get(json_url, timeout=15)
            
            if response.status_code == 200:
                data = response.json()
                details = {}
                
                # Extract coordinates from reprPoint
                if data.get('reprPoint'):
                    coords = data['reprPoint']
                    if len(coords) >= 2:
                        details['geo_bounds'] = {
                            'centroid': {
                                'lat': float(coords[1]),  # reprPoint is [lon, lat]
                                'lon': float(coords[0])
                            }
                        }
                        print(f"Found coordinates: {coords[1]}, {coords[0]}")
                
                # Extract place types
                if data.get('placeTypes'):
                    place_types = []
                    for pt in data['placeTypes']:
                        if isinstance(pt, dict) and pt.get('title'):
                            place_types.append(pt['title'])
                    if place_types:
                        details['place_types'] = place_types
                
                # Extract temporal information (simplified)
                if data.get('attestations') or data.get('connectsWith'):
                    details['temporal_bounds'] = 'Ancient period'
                
                return details
            else:
                print(f"Failed to get Pleiades JSON: {response.status_code}")
                
        except Exception as e:
            print(f"Error getting Pleiades JSON for {pleiades_id}: {e}")
        
        return None
    
    def enhance_entities_with_pelagios(self, entities: List[Dict], context_info: Dict = None) -> List[Dict]:
        """
        Enhanced entity enrichment with context-aware searching.
        
        Args:
            entities: List of entities to enhance
            context_info: Detected context information (regions, time periods, etc.)
        """
        print(f"\nStarting Pelagios enhancement for {len(entities)} entities")
        if context_info:
            print(f"Using context: {json.dumps(context_info, indent=2)}")
        
        # Focus on place-like entities
        place_types = ['GPE', 'LOCATION', 'FACILITY']
        enhanced_count = 0
        
        for entity in entities:
            entity_text = entity['text'].strip()
            entity_type = entity['type']
            
            print(f"\nProcessing: '{entity_text}' (type: {entity_type})")
            
            if entity_type in place_types and len(entity_text) > 2:
                # Skip obvious non-places
                skip_terms = ['the', 'and', 'or', 'of', 'in', 'to', 'for', 'with', 'at', 'by']
                if entity_text.lower() in skip_terms:
                    print(f"Skipping common word: '{entity_text}'")
                    continue
                
                # Build context-aware search terms
                search_terms = [entity_text]
                
                # Add context-based variations
                if context_info:
                    # Add region-specific searches
                    if context_info.get('regions'):
                        for region in context_info['regions'][:2]:
                            search_terms.append(f"{entity_text} {region}")
                    
                    # Add time-period hints
                    if context_info.get('historical_era'):
                        search_terms.append(f"{entity_text} {context_info['historical_era']}")
                    
                    # Add civilization-specific searches
                    if context_info.get('ancient_civilizations'):
                        for civ in context_info['ancient_civilizations'][:1]:
                            if 'Greece' in civ:
                                search_terms.append(f"{entity_text} ancient Greece")
                            elif 'Rome' in civ:
                                search_terms.append(f"{entity_text} Roman")
                            elif 'Egypt' in civ:
                                search_terms.append(f"{entity_text} Egypt")
                
                # Try each search term
                results = []
                for search_term in search_terms:
                    print(f"Trying search term: '{search_term}'")
                    
                    # Strategy 1: Try Pleiades direct search
                    results = self.search_pleiades_directly(search_term)
                    
                    # Strategy 2: Try Peripleo legacy API if Pleiades fails
                    if not results:
                        results = self.search_peripleo_legacy(search_term)
                    
                    if results:
                        print(f"Found results with term: '{search_term}'")
                        break
                
                # Process results
                if results:
                    # If we have context, try to select the best match
                    best_match = self._select_best_match_with_context(results, entity_text, context_info)
                    if not best_match:
                        best_match = results[0]
                    
                    print(f"Found match: {best_match.get('title', 'No title')}")
                    
                    # Store Pelagios data
                    entity['pelagios_data'] = {
                        'peripleo_id': best_match.get('identifier'),
                        'peripleo_title': best_match.get('title'),
                        'peripleo_description': best_match.get('description'),
                        'temporal_bounds': best_match.get('temporal_bounds'),
                        'place_types': best_match.get('place_types', []),
                        'peripleo_url': best_match.get('identifier'),
                        'search_context': context_info
                    }
                    
                    # Extract coordinates
                    self._extract_coordinates(entity, best_match)
                    
                    # Extract Pleiades ID if available
                    pleiades_id = self._extract_pleiades_id(best_match)
                    if pleiades_id:
                        entity['pleiades_id'] = pleiades_id
                        entity['pleiades_url'] = f"{self.pleiades_base_url}/places/{pleiades_id}"
                        print(f"Added Pleiades link: {entity['pleiades_url']}")
                    
                    enhanced_count += 1
                else:
                    print(f"No results found for '{entity_text}' even with context")
                
                # Rate limiting
                time.sleep(1.0)  # Be nice to the APIs
        
        print(f"\nPelagios enhancement complete! Enhanced {enhanced_count}/{len(entities)} entities")
        return entities
    
    def _select_best_match_with_context(self, results: List[Dict], entity_text: str, 
                                       context_info: Dict) -> Optional[Dict]:
        """
        Select the best match from results based on context information.
        """
        if not context_info or not results:
            return None
        
        scored_results = []
        
        for result in results:
            score = 0
            
            # Check title match
            if entity_text.lower() in result.get('title', '').lower():
                score += 10
            
            # Check temporal alignment
            if context_info.get('time_period') and result.get('temporal_bounds'):
                result_period = result['temporal_bounds'].lower()
                context_period = context_info['time_period'].lower()
                
                # Check for BCE/CE alignment
                if 'bce' in context_period and 'bc' in result_period:
                    score += 5
                elif 'ce' in context_period and 'ad' in result_period:
                    score += 5
                
                # Check for specific era matches
                if context_info.get('historical_era'):
                    era = context_info['historical_era'].lower()
                    if any(word in result_period for word in era.split()):
                        score += 8
            
            # Check geographical alignment
            description = result.get('description', '').lower()
            title = result.get('title', '').lower()
            
            if context_info.get('regions'):
                for region in context_info['regions']:
                    region_lower = region.lower()
                    if region_lower in description or region_lower in title:
                        score += 7
            
            if context_info.get('ancient_civilizations'):
                for civ in context_info['ancient_civilizations']:
                    civ_keywords = {
                        'Ancient Greece': ['greek', 'hellas', 'hellenic'],
                        'Roman Empire': ['roman', 'rome'],
                        'Ancient Egypt': ['egypt', 'egyptian'],
                        'Persian Empire': ['persian', 'achaemenid']
                    }
                    
                    keywords = civ_keywords.get(civ, [civ.lower()])
                    if any(kw in description or kw in title for kw in keywords):
                        score += 6
            
            # Check place type relevance
            place_types = result.get('place_types', [])
            if place_types:
                relevant_types = ['settlement', 'city', 'town', 'port', 'fortress']
                if any(pt.lower() in relevant_types for pt in place_types):
                    score += 3
            
            scored_results.append((score, result))
        
        # Sort by score and return the best match
        scored_results.sort(key=lambda x: x[0], reverse=True)
        
        if scored_results and scored_results[0][0] > 0:
            return scored_results[0][1]
        
        return None
    
    def _extract_coordinates(self, entity: Dict, match_data: Dict):
        """Extract coordinates from Pelagios data."""
        # Look for geo_bounds with centroid
        geo_bounds = match_data.get('geo_bounds')
        if geo_bounds and isinstance(geo_bounds, dict):
            centroid = geo_bounds.get('centroid')
            if centroid and isinstance(centroid, dict):
                lat = centroid.get('lat')
                lon = centroid.get('lon')
                if lat is not None and lon is not None:
                    entity['latitude'] = float(lat)
                    entity['longitude'] = float(lon)
                    entity['geocoding_source'] = 'pelagios'
                    print(f"Added coordinates: {lat}, {lon}")
                    return
        
        # Fallback: look for other coordinate formats
        for coord_key in ['coordinates', 'location', 'geometry']:
            coords = match_data.get(coord_key)
            if coords and isinstance(coords, (list, tuple)) and len(coords) >= 2:
                try:
                    entity['longitude'] = float(coords[0])
                    entity['latitude'] = float(coords[1])
                    entity['geocoding_source'] = 'pelagios'
                    print(f"Added coordinates: {coords[1]}, {coords[0]}")
                    return
                except (ValueError, TypeError):
                    continue
    
    def _extract_pleiades_id(self, match_data: Dict) -> Optional[str]:
        """Extract Pleiades ID from match data."""
        # Direct source_id for Pleiades items
        if match_data.get('source_gazetteer') == 'pleiades':
            source_id = match_data.get('source_id')
            if source_id:
                return str(source_id)
        
        # Extract from identifier/URL
        identifier = match_data.get('identifier', '')
        if 'pleiades.stoa.org/places/' in identifier:
            try:
                pleiades_id = identifier.split('/places/')[-1].split('/')[0].split('?')[0]
                if pleiades_id.isdigit():
                    return pleiades_id
            except:
                pass
        
        return None

    # Keep the export methods exactly the same as before...
    def export_to_recogito_format(self, text: str, entities: List[Dict], 
                                 title: str = "EntityLinker Export") -> str:
        """Export entities in Recogito-compatible JSON format."""
        annotations = []
        
        for idx, entity in enumerate(entities):
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
        """Export annotated text as TEI XML with place markup."""
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
        
        # Add XML declaration
        pretty_xml = '<?xml version="1.0" encoding="UTF-8"?>\n' + xml_str
        
        return pretty_xml
    
    def _create_tei_markup(self, text: str, entities: List[Dict], tei_ns: str):
        """Create TEI markup with place tags."""
        sorted_entities = sorted(entities, key=lambda x: x['start'])
        
        p = ET.Element("{%s}p" % tei_ns)
        
        current_pos = 0
        
        for entity in sorted_entities:
            if entity['start'] > current_pos:
                if len(p) == 0 and p.text is None:
                    p.text = text[current_pos:entity['start']]
                else:
                    if len(p) > 0:
                        if p[-1].tail is None:
                            p[-1].tail = text[current_pos:entity['start']]
                        else:
                            p[-1].tail += text[current_pos:entity['start']]
            
            if entity['type'] in ['GPE', 'LOCATION']:
                place_elem = ET.SubElement(p, "{%s}placeName" % tei_ns)
            elif entity['type'] == 'PERSON':
                place_elem = ET.SubElement(p, "{%s}persName" % tei_ns)
            elif entity['type'] == 'ORGANIZATION':
                place_elem = ET.SubElement(p, "{%s}orgName" % tei_ns)
            else:
                place_elem = ET.SubElement(p, "{%s}name" % tei_ns)
            
            place_elem.text = entity['text']
            
            if entity.get('pleiades_id'):
                place_elem.set('ref', entity['pleiades_url'])
            elif entity.get('wikidata_url'):
                place_elem.set('ref', entity['wikidata_url'])
            
            if entity.get('latitude') and entity.get('longitude'):
                place_elem.set('geo', f"{entity['latitude']},{entity['longitude']}")
            
            current_pos = entity['end']
        
        if current_pos < len(text):
            if len(p) == 0:
                p.text = (p.text or "") + text[current_pos:]
            else:
                p[-1].tail = (p[-1].tail or "") + text[current_pos:]
        
        return p
    
    def create_pelagios_map_url(self, entities: List[Dict]) -> str:
        """Create a Peripleo map URL showing all georeferenced entities."""
        geo_entities = [e for e in entities if e.get('latitude') and e.get('longitude')]
        
        if not geo_entities:
            return "https://peripleo.pelagios.org"
        
        lats = [e['latitude'] for e in geo_entities]
        lons = [e['longitude'] for e in geo_entities]
        
        min_lat, max_lat = min(lats), max(lats)
        min_lon, max_lon = min(lons), max(lons)
        
        lat_padding = (max_lat - min_lat) * 0.1 or 0.01
        lon_padding = (max_lon - min_lon) * 0.1 or 0.01
        
        bbox = f"{min_lon-lon_padding},{min_lat-lat_padding},{max_lon+lon_padding},{max_lat+lat_padding}"
        
        return f"https://peripleo.pelagios.org/ui#bbox={bbox}"


class EntityLinker:
    """
    Main class for entity linking functionality with improved Pelagios integration.
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
            'ADDRESS': '#CCBEAA',         # F&B Oxford stone
            'NORP': '#D4C5B9'             # F&B Elephant's breath light
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
        """Extract named entities from text using spaCy with improved validation."""
        # Process text with spaCy
        doc = self.nlp(text)
        
        entities = []
        
        # Step 1: Extract traditional named entities with enhanced validation
        for ent in doc.ents:
            # Filter out unwanted entity types at the spaCy label level
            if ent.label_ in ['DATE', 'TIME', 'PERCENT', 'MONEY', 'QUANTITY', 'ORDINAL', 'CARDINAL']:
                continue
            
            # Map spaCy entity types to our format
            entity_type = self._map_spacy_entity_type(ent.label_)
            
            # Additional filter in case mapping returns an unwanted type
            if entity_type in ['DATE', 'TIME', 'PERCENT', 'MONEY', 'QUANTITY', 'ORDINAL', 'CARDINAL']:
                continue
            
            # Enhanced validation using grammatical context
            if self._is_valid_entity(ent.text, entity_type, ent):
                # Fix: spaCy Span objects don't have .get() method
                # Instead, check if the span has a confidence attribute or use default
                confidence = 1.0
                if hasattr(ent, '_') and hasattr(ent._, 'confidence'):
                    confidence = ent._.confidence
                elif hasattr(ent, 'ent_score'):
                    confidence = ent.ent_score
                
                entities.append({
                    'text': ent.text,
                    'type': entity_type,
                    'start': ent.start_char,
                    'end': ent.end_char,
                    'label': ent.label_,  # Keep original spaCy label
                    'confidence': confidence
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
            'NORP': 'NORP',  # Nationalities or religious or political groups
            'EVENT': 'LOCATION',  # Events often have location relevance
            'WORK_OF_ART': 'ORGANIZATION',  # Often associated with organizations
            'LAW': 'ORGANIZATION',  # Laws often associated with organizations
            'LANGUAGE': 'GPE'  # Languages associated with places
        }
        return mapping.get(spacy_label, spacy_label)

    def _is_valid_entity(self, entity_text: str, entity_type: str, spacy_ent) -> bool:
        """Enhanced entity validation using spaCy's linguistic features."""
        # Skip very short entities
        if len(entity_text.strip()) <= 1:
            return False
        
        # Skip common stopwords
        if entity_text.lower() in ['the', 'and', 'or', 'of', 'in', 'to', 'for', 'with', 'at', 'by']:
            return False
        
        # Get the token(s) for this entity
        doc = spacy_ent.doc
        entity_tokens = [token for token in doc[spacy_ent.start:spacy_ent.end]]
        
        if not entity_tokens:
            return True  # Default to valid if we can't analyze
        
        first_token = entity_tokens[0]
        
        # Filter out words functioning primarily as verbs or adjectives
        if first_token.pos_ in ['VERB', 'AUX']:
            return False
        
        # Be more selective with adjectives for person entities
        if first_token.pos_ == 'ADJ' and entity_type == 'PERSON':
            return False
        
        # Prefer proper nouns as they're more likely to be real entities
        if first_token.pos_ == 'PROPN':
            return True
        
        # For place names, be more lenient with nouns
        if entity_type in ['GPE', 'LOCATION', 'FACILITY'] and first_token.pos_ == 'NOUN':
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
                    'end': match.end(),
                    'confidence': 0.8
                })
        
        return addresses

    def _remove_overlapping_entities(self, entities):
        """Remove overlapping entities, keeping the longest and highest confidence ones."""
        entities.sort(key=lambda x: (x['start'], -len(x['text']), -x.get('confidence', 1.0)))
        
        filtered = []
        for entity in entities:
            overlaps = False
            for existing in filtered[:]:  # Create a copy to safely modify during iteration
                # Check if entities overlap
                if (entity['start'] < existing['end'] and entity['end'] > existing['start']):
                    # Compare by length first, then confidence
                    entity_score = len(entity['text']) + entity.get('confidence', 1.0)
                    existing_score = len(existing['text']) + existing.get('confidence', 1.0)
                    
                    if entity_score > existing_score:
                        filtered.remove(existing)
                        break
                    else:
                        # Current entity is lower quality, skip it
                        overlaps = True
                        break
            
            if not overlaps:
                filtered.append(entity)
        
        return filtered

    def _detect_geographical_context(self, text: str, entities: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Enhanced context detection using multiple free LLM options (Ollama, HuggingFace, OpenRouter).
        Returns geographical and temporal context to improve entity linking accuracy.
        """
        context = {
            'regions': [],
            'time_period': None,
            'historical_era': None,
            'modern_countries': [],
            'ancient_civilizations': []
        }
        
        # Try multiple LLM providers in order of preference
        providers = [
            self._detect_context_ollama,
            self._detect_context_huggingface,
            self._detect_context_openrouter,
            self._detect_context_regex_fallback
        ]
        
        for provider in providers:
            try:
                detected_context = provider(text[:2000])  # Use first 2000 chars
                if detected_context and any(detected_context.values()):
                    context.update(detected_context)
                    print(f"Context detected: {context}")
                    return context
            except Exception as e:
                print(f"Context detection failed with {provider.__name__}: {e}")
                continue
        
        # If all fail, use simple regex patterns
        return self._detect_context_regex_fallback(text)
    
    def _detect_context_ollama(self, text: str) -> Dict[str, Any]:
        """
        Use Ollama (local LLM) for context detection - most reliable for offline use.
        Requires Ollama to be installed and running locally.
        """
        try:
            import requests
            import json
            
            # Check if Ollama is running
            try:
                response = requests.get('http://localhost:11434/api/tags', timeout=2)
                if response.status_code != 200:
                    raise Exception("Ollama not responding")
            except:
                raise Exception("Ollama not running on localhost:11434")
            
            # Prepare enhanced prompt for better context extraction
            prompt = f"""Analyze this historical text and extract geographical and temporal context. Return a JSON object with these fields:
- regions: list of geographical regions mentioned or implied (e.g., "Mediterranean", "Near East", "Greece")
- modern_countries: list of modern country names that correspond to places mentioned
- ancient_civilizations: list of ancient civilizations or empires mentioned (e.g., "Persian Empire", "Ancient Greece")
- time_period: approximate time period in years (e.g., "500-400 BCE", "1st century CE")
- historical_era: named historical period (e.g., "Classical Antiquity", "Hellenistic Period")

Text: {text}

Respond ONLY with valid JSON, no other text."""

            # Try different models in order of preference
            models = ['mistral', 'llama2', 'neural-chat', 'vicuna']
            
            for model in models:
                try:
                    response = requests.post(
                        'http://localhost:11434/api/generate',
                        json={
                            'model': model,
                            'prompt': prompt,
                            'stream': False,
                            'options': {
                                'temperature': 0.3,
                                'top_p': 0.9
                            }
                        },
                        timeout=30
                    )
                    
                    if response.status_code == 200:
                        result = response.json()
                        response_text = result.get('response', '')
                        
                        # Extract JSON from response
                        import re
                        json_match = re.search(r'\{.*\}', response_text, re.DOTALL)
                        if json_match:
                            context_data = json.loads(json_match.group())
                            return context_data
                except Exception as e:
                    print(f"Ollama model {model} failed: {e}")
                    continue
                    
        except Exception as e:
            print(f"Ollama context detection failed: {e}")
            
        return {}
    
    def _detect_context_huggingface(self, text: str) -> Dict[str, Any]:
        """
        Use HuggingFace Inference API with free models for context detection.
        Requires HF_TOKEN environment variable or uses free tier limits.
        """
        try:
            import requests
            import os
            
            # Use free inference API
            api_url = "https://api-inference.huggingface.co/models/mistralai/Mistral-7B-Instruct-v0.1"
            headers = {}
            
            # Optional: use HF token if available for higher rate limits
            hf_token = os.environ.get('HF_TOKEN')
            if hf_token:
                headers["Authorization"] = f"Bearer {hf_token}"
            
            prompt = f"""Extract geographical and temporal context from this text.
Return regions, countries, time periods, and civilizations mentioned.

Text: {text[:500]}

Context:"""
            
            response = requests.post(
                api_url,
                headers=headers,
                json={"inputs": prompt, "parameters": {"max_new_tokens": 200, "temperature": 0.3}},
                timeout=20
            )
            
            if response.status_code == 200:
                result = response.json()
                if isinstance(result, list) and result:
                    generated_text = result[0].get('generated_text', '')
                    # Parse the response to extract context
                    return self._parse_llm_context_response(generated_text)
                    
        except Exception as e:
            print(f"HuggingFace API failed: {e}")
            
        return {}
    
    def _detect_context_openrouter(self, text: str) -> Dict[str, Any]:
        """
        Use OpenRouter API with free models for context detection.
        Some models available without API key.
        """
        try:
            import requests
            import os
            
            # OpenRouter API with free models
            api_url = "https://openrouter.ai/api/v1/chat/completions"
            
            # Optional API key for better models
            api_key = os.environ.get('OPENROUTER_API_KEY')
            headers = {
                "Content-Type": "application/json",
                "HTTP-Referer": "https://github.com/entity-linker",
                "X-Title": "EntityLinker"
            }
            
            if api_key:
                headers["Authorization"] = f"Bearer {api_key}"
            
            # Use a free model
            model = "mistralai/mistral-7b-instruct:free" if not api_key else "mistralai/mistral-7b-instruct"
            
            messages = [{
                "role": "system",
                "content": "You are a historical geography expert. Extract geographical and temporal context from texts."
            }, {
                "role": "user", 
                "content": f"From this text, identify: 1) geographical regions 2) time period 3) civilizations. Text: {text[:500]}"
            }]
            
            response = requests.post(
                api_url,
                headers=headers,
                json={
                    "model": model,
                    "messages": messages,
                    "temperature": 0.3
                },
                timeout=20
            )
            
            if response.status_code == 200:
                result = response.json()
                content = result['choices'][0]['message']['content']
                return self._parse_llm_context_response(content)
                
        except Exception as e:
            print(f"OpenRouter API failed: {e}")
            
        return {}
    
    def _detect_context_regex_fallback(self, text: str) -> Dict[str, Any]:
        """
        Fallback context detection using regex patterns and keyword matching.
        Works offline without any LLM dependencies.
        """
        import re
        
        context = {
            'regions': [],
            'modern_countries': [],
            'ancient_civilizations': [],
            'time_period': None,
            'historical_era': None
        }
        
        # Enhanced region patterns
        region_patterns = {
            'Mediterranean': r'\b(Mediterranean|Aegean|Adriatic|Ionian)\b',
            'Near East': r'\b(Mesopotamia|Babylon|Assyria|Syria|Levant|Anatolia)\b',
            'Greece': r'\b(Greece|Hellas|Hellenic|Greek|Attica|Peloponnese)\b',
            'Egypt': r'\b(Egypt|Egyptian|Nile|Memphis|Thebes|Alexandria)\b',
            'Persia': r'\b(Persia|Persian|Iran|Achaemenid|Parthia)\b',
            'Rome': r'\b(Rome|Roman|Italia|Latin|Republic|Empire)\b',
            'Asia Minor': r'\b(Asia Minor|Anatolia|Phrygia|Lydia|Caria)\b'
        }
        
        # Check for regions
        for region, pattern in region_patterns.items():
            if re.search(pattern, text, re.IGNORECASE):
                context['regions'].append(region)
        
        # Time period detection
        century_match = re.search(r'\b(\d+)(?:st|nd|rd|th)\s+century\s+(BCE?|CE|AD|BC)\b', text, re.IGNORECASE)
        if century_match:
            century = int(century_match.group(1))
            era = century_match.group(2).upper()
            if era in ['BC', 'BCE']:
                context['time_period'] = f"{century}00-{(century-1)}00 BCE"
            else:
                context['time_period'] = f"{(century-1)}00-{century}00 CE"
        
        # Year range detection
        year_range = re.search(r'\b(\d{1,4})\s*[-–]\s*(\d{1,4})\s*(BCE?|CE|AD|BC)\b', text, re.IGNORECASE)
        if year_range and not context['time_period']:
            start, end, era = year_range.groups()
            context['time_period'] = f"{start}-{end} {era.upper()}"
        
        # Historical era detection
        era_keywords = {
            'Classical Antiquity': ['classical', 'antiquity', 'ancient world'],
            'Hellenistic Period': ['hellenistic', 'successor', 'diadochi'],
            'Roman Period': ['roman empire', 'roman republic', 'imperial'],
            'Bronze Age': ['bronze age', 'minoan', 'mycenaean'],
            'Iron Age': ['iron age'],
            'Archaic Period': ['archaic', 'early greek']
        }
        
        text_lower = text.lower()
        for era, keywords in era_keywords.items():
            if any(keyword in text_lower for keyword in keywords):
                context['historical_era'] = era
                break
        
        # Ancient civilizations
        civ_patterns = {
            'Ancient Greece': r'\b(Greek|Hellenic|Athens|Sparta)\b',
            'Persian Empire': r'\b(Persian|Achaemenid|Xerxes|Darius)\b',
            'Roman Empire': r'\b(Roman|Rome|Caesar|Augustus)\b',
            'Ancient Egypt': r'\b(Egypt|Pharaoh|pyramid)\b',
            'Phoenicia': r'\b(Phoenician|Carthage|Tyre|Sidon)\b',
            'Assyria': r'\b(Assyrian|Nineveh|Ashur)\b',
            'Babylon': r'\b(Babylon|Nebuchadnezzar|Hammurabi)\b'
        }
        
        for civ, pattern in civ_patterns.items():
            if re.search(pattern, text, re.IGNORECASE):
                context['ancient_civilizations'].append(civ)
        
        # Remove duplicates
        context['regions'] = list(set(context['regions']))
        context['ancient_civilizations'] = list(set(context['ancient_civilizations']))
        
        return context
    
    def _parse_llm_context_response(self, response_text: str) -> Dict[str, Any]:
        """Parse LLM response to extract context information."""
        import re
        import json
        
        context = {
            'regions': [],
            'modern_countries': [],
            'ancient_civilizations': [],
            'time_period': None,
            'historical_era': None
        }
        
        # Try to parse as JSON first
        try:
            json_match = re.search(r'\{.*\}', response_text, re.DOTALL)
            if json_match:
                parsed = json.loads(json_match.group())
                return {k: v for k, v in parsed.items() if k in context}
        except:
            pass
        
        # Fallback to text parsing
        lines = response_text.split('\n')
        for line in lines:
            line_lower = line.lower()
            
            # Extract regions
            if 'region' in line_lower:
                regions = re.findall(r':\s*(.+)', line)
                if regions:
                    context['regions'] = [r.strip() for r in regions[0].split(',')]
            
            # Extract time period
            if 'period' in line_lower or 'century' in line_lower:
                period_match = re.search(r'(\d+.*(?:BCE?|CE|AD|BC))', line, re.IGNORECASE)
                if period_match:
                    context['time_period'] = period_match.group(1)
            
            # Extract civilizations
            if 'civiliz' in line_lower:
                civs = re.findall(r':\s*(.+)', line)
                if civs:
                    context['ancient_civilizations'] = [c.strip() for c in civs[0].split(',')]
        
        return context

    def get_coordinates(self, entities):
        """Enhanced coordinate lookup with comprehensive geographical context detection."""
        import requests
        import time
        
        # Get the full text from session state if available
        full_text = st.session_state.get('processed_text', '')
        
        # Detect comprehensive geographical and temporal context
        context_info = self._detect_geographical_context(full_text, entities)
        
        # Log detected context for debugging
        if st.session_state.get('show_debug', False):
            print(f"Detected context: {json.dumps(context_info, indent=2)}")
        
        place_types = ['GPE', 'LOCATION', 'FACILITY', 'ORGANIZATION', 'ADDRESS', 'NORP']
        
        for entity in entities:
            if entity['type'] in place_types:
                # Skip if already has coordinates from Pelagios
                if entity.get('latitude') is not None:
                    continue
                
                # Try geocoding with enhanced context
                if self._try_contextual_geocoding(entity, context_info):
                    continue
                    
                # Fall back to original methods
                if self._try_python_geocoding(entity):
                    continue
                
                if self._try_openstreetmap(entity):
                    continue
                    
                # If still no coordinates, try aggressive geocoding with context
                self._try_aggressive_geocoding_with_context(entity, context_info)
        
        return entities
    
    def _try_contextual_geocoding(self, entity, context_info):
        """
        Enhanced contextual geocoding using detected context for better accuracy.
        Prioritizes historical context when dealing with ancient places.
        """
        import time
        from geopy.geocoders import Nominatim
        from geopy.exc import GeocoderTimedOut, GeocoderServiceError
        
        # Build contextual search terms based on detected context
        search_terms = [entity['text']]
        
        # Add region-specific search terms
        if context_info.get('regions'):
            for region in context_info['regions'][:2]:  # Use top 2 regions
                search_terms.append(f"{entity['text']}, {region}")
        
        # Add civilization-specific terms for ancient places
        if context_info.get('ancient_civilizations'):
            for civ in context_info['ancient_civilizations'][:1]:
                if 'Ancient Greece' in civ:
                    search_terms.append(f"{entity['text']}, Greece")
                    search_terms.append(f"ancient {entity['text']}, Greece")
                elif 'Persian' in civ:
                    search_terms.append(f"{entity['text']}, Iran")
                    search_terms.append(f"ancient {entity['text']}, Persia")
                elif 'Roman' in civ:
                    search_terms.append(f"{entity['text']}, Italy")
                    search_terms.append(f"ancient {entity['text']}")
                elif 'Egypt' in civ:
                    search_terms.append(f"{entity['text']}, Egypt")
                    search_terms.append(f"ancient {entity['text']}, Egypt")
        
        # Add time period hints
        if context_info.get('time_period') and 'BCE' in str(context_info.get('time_period', '')):
            search_terms.append(f"ancient {entity['text']}")
            search_terms.append(f"{entity['text']} archaeological site")
        
        # Try Ollama if available for intelligent disambiguation
        try:
            import requests
            
            # Check if Ollama is available
            response = requests.get('http://localhost:11434/api/tags', timeout=1)
            if response.status_code == 200:
                return self._try_ollama_contextual_geocoding(entity, context_info, search_terms)
        except:
            pass
        
        # Fallback to regular geocoding with context-informed search terms
        geolocator = Nominatim(user_agent="EntityLinker/2.1-Contextual", timeout=10)
        
        for search_term in search_terms:
            try:
                location = geolocator.geocode(search_term, timeout=10)
                if location:
                    entity['latitude'] = location.latitude
                    entity['longitude'] = location.longitude
                    entity['location_name'] = location.address
                    entity['geocoding_source'] = 'geopy_contextual'
                    entity['search_term_used'] = search_term
                    entity['context_use
