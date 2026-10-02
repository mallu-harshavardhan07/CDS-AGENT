*&---------------------------------------------------------------------*
*& Report Z_EXPORT_CDS_CATALOG
*&---------------------------------------------------------------------*
*& Purpose: Extracts active SAP S/4HANA Core Data Services (CDS) view
*&          definitions, descriptions, annotations, and DDL source code
*&          into a standardized JSON catalog (cds_catalog.json) for
*&          downstream RAG search and AI vector indexing.
*& Author:  SAP S/4HANA Cloud & AI Integration Engineering
*& System:  SAP NetWeaver 7.50+ / SAP S/4HANA 1610 - 2023+
*&---------------------------------------------------------------------*
REPORT z_export_cds_catalog.

TABLES: ddddlsrc, ddddlsrc02t, ddddlsrct.

*----------------------------------------------------------------------*
* TYPE DEFINITIONS (Matching Target JSON Schema)
*----------------------------------------------------------------------*
TYPES:
  BEGIN OF ty_cds_entry,
    view_name       TYPE string,
    ddl_source_name TYPE string,
    description     TYPE string,
    annotations     TYPE string_table,
    ddl_code        TYPE string,
  END OF ty_cds_entry,
  tt_cds_catalog TYPE STANDARD TABLE OF ty_cds_entry WITH DEFAULT KEY.

TYPES:
  BEGIN OF ty_ddl_meta,
    ddlname   TYPE ddddlsrc-ddlname,
    parentname TYPE ddddlsrc-parentname,
    source    TYPE string,
    strucobjn TYPE ddddlsrc02t-strucobjn,
    ddtext    TYPE ddddlsrct-ddtext,
  END OF ty_ddl_meta,
  tt_ddl_meta TYPE STANDARD TABLE OF ty_ddl_meta WITH DEFAULT KEY.

*----------------------------------------------------------------------*
* SELECTION SCREEN
*----------------------------------------------------------------------*
SELECTION-SCREEN BEGIN OF BLOCK b01 WITH FRAME TITLE TEXT-b01.
  SELECT-OPTIONS: s_ddl  FOR ddddlsrc-ddlname DEFAULT 'I_*' OPTION CP SIGN I.
  PARAMETERS:     p_lang TYPE sy-langu DEFAULT 'E' OBLIGATORY,
                  p_max  TYPE i DEFAULT 500,
                  p_annos AS CHECKBOX DEFAULT 'X'.
SELECTION-SCREEN END OF BLOCK b01.

SELECTION-SCREEN BEGIN OF BLOCK b02 WITH FRAME TITLE TEXT-b02.
  PARAMETERS: p_file TYPE string DEFAULT 'C:\temp\cds_catalog.json' LOWER CASE,
              p_down AS CHECKBOX DEFAULT 'X',
              p_disp AS CHECKBOX DEFAULT ' '.
SELECTION-SCREEN END OF BLOCK b02.

*----------------------------------------------------------------------*
* INITIALIZATION & VALUE HELP
*----------------------------------------------------------------------*
INITIALIZATION.
  TEXT-b01 = 'CDS Extraction Scope'.
  TEXT-b02 = 'Export & Download Options'.

AT SELECTION-SCREEN ON VALUE-REQUEST FOR p_file.
  DATA: lt_file_table TYPE filetable,
        lv_rc         TYPE i,
        lv_action     TYPE i,
        lv_default_path TYPE string,
        lv_filename   TYPE string VALUE 'cds_catalog.json',
        lv_path       TYPE string,
        lv_fullpath   TYPE string.

  cl_gui_frontend_services=>file_save_dialog(
    EXPORTING
      window_title         = 'Save CDS Catalog JSON'
      default_extension    = 'json'
      default_file_name    = lv_filename
      file_filter          = 'JSON Files (*.json)|*.json|All Files (*.*)|*.*'
    CHANGING
      filename             = lv_filename
      path                 = lv_path
      fullpath             = lv_fullpath
      user_action          = lv_action
    EXCEPTIONS
      OTHERS               = 1
  ).
  IF sy-subrc = 0 AND lv_action <> cl_gui_frontend_services=>action_cancel.
    p_file = lv_fullpath.
  ENDIF.

*----------------------------------------------------------------------*
* CLASS DEFINITION: lcl_cds_extractor
*----------------------------------------------------------------------*
CLASS lcl_cds_extractor DEFINITION FINAL.
  PUBLIC SECTION.
    METHODS:
      extract_catalog
        RETURNING
          VALUE(rt_catalog) TYPE tt_cds_catalog,
      serialize_to_json
        IMPORTING
          it_catalog     TYPE tt_cds_catalog
        RETURNING
          VALUE(rv_json) TYPE string,
      download_file
        IMPORTING
          iv_filepath TYPE string
          iv_content  TYPE string
        RETURNING
          VALUE(rv_ok) TYPE abap_bool,
      display_summary
        IMPORTING
          it_catalog TYPE tt_cds_catalog.

  PRIVATE SECTION.
    METHODS:
      extract_key_annotations
        IMPORTING
          iv_ddl_code       TYPE string
        RETURNING
          VALUE(rt_annos)   TYPE string_table,
      clean_ddl_text
        IMPORTING
          iv_raw_source     TYPE string
        RETURNING
          VALUE(rv_clean)   TYPE string,
      get_view_description
        IMPORTING
          iv_ddlname        TYPE ddddlsrc-ddlname
          iv_strucobjn      TYPE ddddlsrc02t-strucobjn
        RETURNING
          VALUE(rv_desc)    TYPE string.
ENDCLASS.

*----------------------------------------------------------------------*
* CLASS IMPLEMENTATION: lcl_cds_extractor
*----------------------------------------------------------------------*
CLASS lcl_cds_extractor IMPLEMENTATION.

  METHOD extract_catalog.
    DATA: lt_meta       TYPE tt_ddl_meta,
          ls_entry      TYPE ty_cds_entry,
          lv_limit      TYPE i.

    lv_limit = p_max.
    IF lv_limit <= 0.
      lv_limit = 100000. " Unlimited safe ceiling
    ENDIF.

    WRITE: / '----------------------------------------------------------------------',
           / 'SAP S/4HANA Core Data Services (CDS) Extraction Engine',
           / '----------------------------------------------------------------------'.
    WRITE: / 'Reading DDL sources from SAP Dictionary repository...'.

    " 1. Query active DDL sources and structured CDS view names
    " Joins DDDDLSRC with DDDDLSRC02T (View Name mapping) and DDDDLSRCT (Description)
    SELECT d~ddlname,
           d~parentname,
           d~source,
           t~strucobjn,
           txt~ddtext
      FROM ddddlsrc AS d
      LEFT OUTER JOIN ddddlsrc02t AS t
        ON t~ddlname = d~ddlname
      LEFT OUTER JOIN ddddlsrct AS txt
        ON txt~ddlname    = d~ddlname
       AND txt~ddlanguage = @p_lang
      WHERE d~ddlname IN @s_ddl
        AND d~as4local = 'A' " Active definitions only
      ORDER BY d~ddlname
      INTO CORRESPONDING FIELDS OF TABLE @lt_meta
      UP TO @lv_limit ROWS.

    IF lt_meta IS INITIAL.
      " Fallback: check DDDDLSRC without joins in case 02T/SRCT are partitioned or customized
      SELECT ddlname, parentname, source
        FROM ddddlsrc
        WHERE ddlname IN @s_ddl
          AND as4local = 'A'
        ORDER BY ddlname
        INTO CORRESPONDING FIELDS OF TABLE @lt_meta
        UP TO @lv_limit ROWS.
    ENDIF.

    WRITE: / 'Found', lines( lt_meta ), 'active DDL source records matching criteria.'.

    " 2. Transform into target JSON catalog entities
    LOOP AT lt_meta ASSIGNING FIELD-SYMBOL(<ls_meta>).
      CLEAR ls_entry.

      " Resolve DDL Source Name
      ls_entry-ddl_source_name = <ls_meta>-ddlname.

      " Resolve CDS View Technical Name
      IF <ls_meta>-strucobjn IS NOT INITIAL.
        ls_entry-view_name = <ls_meta>-strucobjn.
      ELSE.
        " Parse 'define [root] view [entity] <view_name>' from DDL source code
        FIND FIRST OCCURRENCE OF REGEX '(?:define\s+(?:root\s+)?(?:view\s+(?:entity\s+)?|table\s+function\s+))([A-Za-z0-9_]+)'
          IN <ls_meta>-source
          SUBMATCHES ls_entry-view_name
          IGNORING CASE.
        IF ls_entry-view_name IS INITIAL.
          ls_entry-view_name = <ls_meta>-ddlname.
        ENDIF.
      ENDIF.

      " Resolve Human-Readable Description
      IF <ls_meta>-ddtext IS NOT INITIAL.
        ls_entry-description = <ls_meta>-ddtext.
      ELSE.
        ls_entry-description = me->get_view_description(
          iv_ddlname   = <ls_meta>-ddlname
          iv_strucobjn = ls_entry-view_name
        ).
      ENDIF.

      " If still no description, attempt extraction from @EndUserText.label
      IF ls_entry-description IS INITIAL.
        FIND FIRST OCCURRENCE OF REGEX '@EndUserText\.label\s*:\s*''([^'']+)'''
          IN <ls_meta>-source
          SUBMATCHES ls_entry-description
          IGNORING CASE.
      ENDIF.

      IF ls_entry-description IS INITIAL.
        ls_entry-description = ls_entry-view_name.
      ENDIF.

      " Full DDL Source Code
      ls_entry-ddl_code = me->clean_ddl_text( <ls_meta>-source ).

      " Extract Key Annotations (@VDM, @Analytics, @EndUserText, @ObjectModel, etc.)
      IF p_annos = abap_true.
        ls_entry-annotations = me->extract_key_annotations( <ls_meta>-source ).
      ENDIF.

      APPEND ls_entry TO rt_catalog.
    ENDLOOP.

  ENDMETHOD.

  METHOD extract_key_annotations.
    " Extracts high-value CDS annotations for semantic search and classification
    DATA: lt_matches TYPE match_result_tab,
          lv_matched TYPE string.

    " Match key annotation patterns e.g. @VDM.viewType: #BASIC, @Analytics.dataCategory: #CUBE
    FIND ALL OCCURRENCES OF REGEX '(@[A-Za-z0-9_]+(?:\.[A-Za-z0-9_]+)+\s*:\s*[^;\r\n\}]+)'
      IN iv_ddl_code
      RESULTS lt_matches
      IGNORING CASE.

    LOOP AT lt_matches ASSIGNING FIELD-SYMBOL(<ls_match>).
      lv_matched = substring( val = iv_ddl_code off = <ls_match>-offset len = <ls_match>-length ).
      CONDENSE lv_matched.

      " Keep critical enterprise annotations
      IF lv_matched CP '@VDM.*' OR
         lv_matched CP '@Analytics.*' OR
         lv_matched CP '@EndUserText.*' OR
         lv_matched CP '@AccessControl.*' OR
         lv_matched CP '@ObjectModel.*' OR
         lv_matched CP '@Semantics.*' OR
         lv_matched CP '@Search.*'.
        APPEND lv_matched TO rt_annos.
      ENDIF.
    ENDLOOP.

    " Ensure annotations list is deduplicated
    SORT rt_annos.
    DELETE ADJACENT DUPLICATES FROM rt_annos.
  ENDMETHOD.

  METHOD clean_ddl_text.
    " Cleans up trailing carriage returns and standardizes line breaks
    rv_clean = iv_raw_source.
    REPLACE ALL OCCURRENCES OF cl_abap_char_utilities=>cr_lf IN rv_clean WITH cl_abap_char_utilities=>newline.
  ENDMETHOD.

  METHOD get_view_description.
    " Queries DD02T for view entity text if DDDDLSRCT is blank
    SELECT SINGLE ddtext
      FROM dd02t
      WHERE tabname    = @iv_strucobjn
        AND ddlanguage = @p_lang
        AND as4local   = 'A'
      INTO @rv_desc.
  ENDMETHOD.

  METHOD serialize_to_json.
    " Uses standard SAP /UI2/CL_JSON serializer for fast, valid JSON output
    rv_json = /ui2/cl_json=>serialize(
      data        = it_catalog
      pretty_name = /ui2/cl_json=>pretty_mode-low_case
      compress    = abap_false
    ).
  ENDMETHOD.

  METHOD download_file.
    DATA: lt_solix TYPE solix_tab,
          lv_size  TYPE i.

    rv_ok = abap_false.

    " Convert JSON string to UTF-8 binary stream
    cl_bcs_convert=>string_to_solix(
      EXPORTING
        iv_string   = iv_content
        iv_codepage = '4110' " UTF-8
      IMPORTING
        et_solix    = lt_solix
        ev_size     = lv_size
    ).

    cl_gui_frontend_services=>gui_download(
      EXPORTING
        bin_filesize = lv_size
        filename     = iv_filepath
        filetype     = 'BIN'
      CHANGING
        data_tab     = lt_solix
      EXCEPTIONS
        OTHERS       = 1
    ).

    IF sy-subrc = 0.
      rv_ok = abap_true.
    ENDIF.
  ENDMETHOD.

  METHOD display_summary.
    WRITE: / '----------------------------------------------------------------------',
           / 'Extraction Summary & Statistics:',
           / '----------------------------------------------------------------------'.
    WRITE: / 'Total CDS Views Processed:', lines( it_catalog ).

    DATA: lv_count TYPE i VALUE 0.
    LOOP AT it_catalog ASSIGNING FIELD-SYMBOL(<ls_view>) UP TO 10.
      lv_count = lv_count + 1.
      WRITE: / lv_count, ')',
             <ls_view>-view_name(30),
             '|', <ls_view>-ddl_source_name(25),
             '|', <ls_view>-description(35).
    ENDLOOP.
    IF lines( it_catalog ) > 10.
      WRITE: / '... and', ( lines( it_catalog ) - 10 ), 'more CDS view entities.'.
    ENDIF.
  ENDMETHOD.

ENDCLASS.

*----------------------------------------------------------------------*
* REPORT MAIN EXECUTION
*----------------------------------------------------------------------*
START-OF-SELECTION.
  DATA(lo_extractor) = NEW lcl_cds_extractor( ).

  " 1. Extract Catalog
  DATA(lt_catalog) = lo_extractor->extract_catalog( ).

  IF lt_catalog IS INITIAL.
    MESSAGE 'No active CDS views found matching the selection criteria.' TYPE 'I'.
    RETURN.
  ENDIF.

  " 2. Display summary in SAP GUI
  lo_extractor->display_summary( lt_catalog ).

  " 3. Serialize to JSON
  WRITE: / 'Serializing CDS catalog into target JSON schema...'.
  DATA(lv_json) = lo_extractor->serialize_to_json( lt_catalog ).
  DATA(lv_bytes) = xstrlen( cl_abap_codepage=>convert_to( source = lv_json codepage = 'UTF-8' ) ).
  WRITE: / 'JSON serialization complete. Total payload size:', lv_bytes, 'bytes.'.

  " 4. Download to file if requested
  IF p_down = abap_true AND p_file IS NOT INITIAL.
    WRITE: / 'Initiating local file download to:', p_file.
    DATA(lv_download_ok) = lo_extractor->download_file(
      iv_filepath = p_file
      iv_content  = lv_json
    ).

    IF lv_download_ok = abap_true.
      WRITE: / 'SUCCESS: File successfully downloaded to local disk:', p_file.
      MESSAGE 'CDS Catalog exported and downloaded successfully.' TYPE 'S'.
    ELSE.
      WRITE: / 'ERROR: GUI file download failed. Check local file path or permissions.'.
      MESSAGE 'File download failed.' TYPE 'E'.
    ENDIF.
  ENDIF.

  " 5. Display raw JSON preview if requested
  IF p_disp = abap_true.
    WRITE: / '----------------------------------------------------------------------',
           / 'JSON Payload Preview (First 500 characters):',
           / '----------------------------------------------------------------------'.
    DATA(lv_preview_len) = nmin( val1 = 500 val2 = strlen( lv_json ) ).
    WRITE: / lv_json(lv_preview_len).
  ENDIF.
