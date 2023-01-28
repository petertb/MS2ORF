#!/usr/bin/env python
# coding: utf-8

import pandas as pd
import numpy as np
from Bio import pairwise2
from Bio.Seq import Seq
from Bio.pairwise2 import format_alignment
from Bio.Seq import translate
import re
import multiprocessing

def read_mgf_headers(filename1, converter_type):
    #-----input: requires pandas library for dataframe assembly, reads in .mgf file from RawConverter or MSConvert
    #-----input: converter_type={'RC' or 'MSC'}
    #-----output: single dataframe containing .mgf headers for each spectrum (excludes peak data)
    
    #---read in .mgf file
    with open(filename1, 'r') as f:
        file1 = f.readlines()
    
    #---create empty lists
    scan_list1 = []
    retention_list1 = []
    charge_list1 = []
    precursor_list1 = []
    
    #---iterate through .mgf file
    for line1 in file1:
        if ("SCANS=" in line1):
            scan_list1.append(line1.split('=')[1].replace('\n', ''))
        if ("scan=" in line1):
            scan_list1.append(line1.split('scan=')[1].replace('"', '').replace('\n', ''))
        if "RTINSECONDS=" in line1:
            retention_list1.append(line1.split('=')[1].replace('\n', ''))
        if "CHARGE=" in line1:
            charge_list1.append(line1.split('=')[1].split('+')[0])
        if "PEPMASS=" in line1:
            if converter_type == 'RC':
                precursor_list1.append(line1.split('=')[1].replace('\n', ''))
            elif converter_type == 'MSC':
                precursor_list1.append(line1.split('=')[1].split(' ')[0])
            else:
                precursor_list1.append(line1.split('=')[1].split(' ')[0])

    #---create dataframe and label columns according to type of converter used (either 'RC' for RawConverter or 'MSC' for MSConvert)
    temp_var1 = 'scan'
    temp_var2 = 'retention'
    temp_var3 = 'charge'
    temp_var4 = 'precursor'
    
    if converter_type == 'RC':
        temp_var1 += '_RC'
        temp_var2 += '_RC'
        temp_var3 += '_RC'
        temp_var4 += '_RC'
    elif converter_type == 'MSC':
        temp_var1 += '_MSC'
        temp_var2 += '_MSC'
        temp_var3 += '_MSC'
        temp_var4 += '_MSC'
    
    local_variables_dict1 = locals()
    
    local_variables_dict1[temp_var1] = scan_list1
    local_variables_dict1[temp_var2] = retention_list1
    local_variables_dict1[temp_var3] = charge_list1
    local_variables_dict1[temp_var4] = precursor_list1
    
    #---format types and round values in dataframe
    temp_df1 = pd.DataFrame([local_variables_dict1[temp_var1], local_variables_dict1[temp_var2], local_variables_dict1[temp_var3], local_variables_dict1[temp_var4]]).transpose()
    temp_df1.columns = [temp_var1, temp_var2, temp_var3, temp_var4]
    temp_df1 = temp_df1.astype({temp_var1 : int, temp_var2 : float, temp_var3 : int, temp_var4 : float})
    temp_df1[[temp_var1, temp_var2, temp_var3, temp_var4]] = temp_df1[[temp_var1, temp_var2, temp_var3, temp_var4]].round(decimals=4)
    
    #---add column for MS2 spectrum position (useful for aligning with Casanovo output later)
    spectrum_position_list1 = []
    temp_count1 = 0
    
    for row1 in temp_df1.itertuples():
        spectrum_position_list1.append(temp_count1)
        temp_count1 += 1
    
    temp_df1['spectrum_ID'] = spectrum_position_list1
    
    #---return dataframe
    return temp_df1

def read_Morpheus(filename1, qval):
    #-----input: Metamorpheus "AllPSMs.psmtsv' file; q-value cut-off for selection of relevant rows
    #-----output: dataframe containing relevant database search columns selected by qvalue cut-off
    
    #---read in tsv file
    temp_df1 = pd.read_csv(filename1, sep='\t')
    
    #---select and rename columns
    temp_df1 = temp_df1[['Scan Number', 'Scan Retention Time', 'Precursor Charge', 'Precursor MZ', 'Peptide Monoisotopic Mass', 'Full Sequence', 'QValue']]
    temp_df1.columns = ['scan', 'retention_MM', 'charge_MM', 'precursor_MM', 'monoisotopic_mass', 'sequence', 'qvalue']
    
    #---set datatypes
    temp_df1 = temp_df1.astype({'scan': int, 'retention_MM': float, 'charge_MM': int, 'precursor_MM': float, 'monoisotopic_mass': str, 'sequence': str, 'qvalue': float})
    
    #---reformat sequence column to remove '|' character and select single sequence where concatenation occurs
    temp_list1 = []

    for row1 in temp_df1.itertuples():
        if '|' in row1.sequence:
            temp_list1.append(row1.sequence.split('|')[0])
        else:
            temp_list1.append(row1.sequence)

    temp_df1['sequence'] = temp_list1
    
    #---choose rows according to qvalue cut-off
    temp_df1 = temp_df1.loc[temp_df1['qvalue'] < qval].copy()
    
    return temp_df1

def read_Casanovo(filename1, suffix_label1):
    #-----input: .csv output from Casanovo
    #-----output: dataframe containing selected columns with column titles containing suffix_label1
    
    #-----read in .csv file
    temp_df1 = pd.read_csv(filename1, sep=',')
    
    #-----rename columns
    temp_var1 = 'spectrum_ID'
    temp_var2 = 'denovo_seq_' + str(suffix_label1)
    temp_var3 = 'peptide_score_' + str(suffix_label1)
    temp_var4 = 'aa_scores_' + str(suffix_label1)
    
    temp_df1.columns = [temp_var1, temp_var2, temp_var3, temp_var4]
    
    return temp_df1

def reformat_Columns_For_Comparison(combined_dataframe1, column_list1):
    #-----input: combined dataframe (.mgf, morpheus, casanovo), list of columns to be reformatted 
    #-----output: combined dataframe containing new reformatted columns for each denovo prediction and each database prediction in preparation for sequence comparison
    
    #-----create empty lists named after selected columns needing reformat
    #-----fill list will data from selected column reformatted using list comprehension to remove PTMs
    local_variables_dict1 = locals()
    temp_dict1 = {}
    
    for temp_count,item1 in enumerate(column_list1):
        temp_dict1[item1] = item1 + '_reformatted'
        temp_array1 = np.array(combined_dataframe1[item1], dtype=str)
        temp_array2 = [s.replace('+15.995', '').replace('+57.021', '').replace('+0.984', '').replace('[Common Variable:Oxidation on M]', '').replace('[Common Fixed:Carbamidomethyl on C]', '').replace('I', 'L') for s in temp_array1]
        local_variables_dict1[temp_dict1[item1]] = temp_array2

    #-----create new columns in copy of input dataframe, then return copied dataframe
    combined_dataframe2 = combined_dataframe1.copy()
    
    for item1 in temp_dict1.keys():
        combined_dataframe2[temp_dict1[item1]] = local_variables_dict1[temp_dict1[item1]]
    
    return combined_dataframe2

def compare_Sequences(combined_dataframe1):
    #-----input: dataframe containing sequences to be compared; sequences need to have PTMs removed and I/L residues need to be combined
    #-----input: columns with "_reformatted" suffix will all be compared pairwise
    #-----output: input dataframe containing new columns with comparison scores
    
    #-----select columns containing '_reformatted' suffix and add to new dataframe
    temp_df1 = pd.DataFrame()
    columns_list1 = []
    
    for item1 in combined_dataframe1.columns:
        if '_reformatted' in item1:
            temp_df1[item1] = combined_dataframe1[item1]
            columns_list1.append(item1)
    
    #-----declare variables for iterating through new dataframe
    temp_seq_list1 = []
    temp_score_list1 = []
    temp_score_list_master1 = []
    
    #-----collect sequences from each row and store in list object, then store row lists in master list
    for row1 in temp_df1.itertuples():
        #---collect all sequences in new dataframe row into a list
        for item1 in range(1, len(row1)):
            temp_seq_list1.append(str(row1[item1]))
        #---iterate through above list and compare sequences
        #---divide comparison scores by self-comparison score to generate a % value
        for count1 in range(0, len(temp_seq_list1)):
            try:
                temp_seq1 = Seq(temp_seq_list1[count1])
                temp_self_score1 = pairwise2.align.globalxx(temp_seq1, temp_seq1)[0][2]
                for count2 in range((count1+1), (len(temp_seq_list1))):
                    try:
                        temp_seq2 = Seq(temp_seq_list1[count2])
                        temp_alignment_result1 = pairwise2.align.globalxx(temp_seq1, temp_seq2)
                        temp_score_list1.append(round((temp_alignment_result1[0][2]/temp_self_score1)*100, 1))
                    except:
                        temp_score_list1.append(0)
            except:
                temp_score_list1.append(0)
        
        #---dump row % scores into master list
        temp_score_list_master1.append(temp_score_list1)
        #---clear non-master lists for next iteration
        temp_score_list1 = []
        temp_seq_list1 = []

    #-----create list of column names for new % score dataframe
    new_column_names1 = []
    for position1,item1 in enumerate(columns_list1):
        temp_name1 = item1.split('_')[-2]
        for position2 in range((position1+1), len(columns_list1)):
            temp_name2 = columns_list1[position2].split('_')[-2]
            temp_name3 = temp_name1 + '_vs_' + temp_name2
            new_column_names1.append(temp_name3)

    #-----create new % score dataframe and name columns based on previous codeblock
    temp_df2 = pd.DataFrame(temp_score_list_master1, columns=new_column_names1)

    #-----merge large input dataframe with new % scores dataframe (on index) and return combined dataframe
    temp_df3 = pd.merge(combined_dataframe1, temp_df2, left_index=True, right_index=True, how='outer').fillna(0)
    return temp_df3

def reformat_aa_Scores(combined_df1, columns_list_to_reformat):
    #-----input: Dataframe containing casanovo output and a list of column names to format
    #-----output: New dataframe with reformatted columns replacing old columns
    
    #-----copy input dataframe so modification doesn't occur on input data
    copied_df1 = combined_df1.copy()
    
    #-----iterate through appropriate columns then each row of input column
    for item1 in columns_list_to_reformat:
        temp_col1 = combined_df1[item1]
        temp_new_col1 = []
        for item2 in temp_col1:
            temp_row1 = item2.replace('[', '').replace(']','')
            temp_list1 = temp_row1.split(', ')
            temp_list1 = [float(i)*100 for i in temp_list1]
            temp_list1 = [round(i, 0) for i in temp_list1]
            temp_list1 = [int(i) for i in temp_list1]
            temp_new_col1.append(temp_list1)
        copied_df1[item1] = temp_new_col1
    
    #-----return modified copied dataframe
    return copied_df1

def mask_denovo_sequence(input_dataframe1, PTM_list1, score_cutoff):
    #-----input: dataframe containing denovo peptides from Casanovo list of PTMs to remove in the format '+#####', and lower score cut-off for masking
    #-----output: denovo sequence with PTMs removed and 'X' in place of low scoring residues
    
    #-----copy dataframe
    temp_df1 = input_dataframe1.copy()
    
    #-----create list of denovo columns
    denovo_column_list1 = []
    
    for item1 in temp_df1.columns:
        if ("denovo_seq" in item1) or ("aa_scores_" in item1):
            if ("_reformatted" not in item1):
                denovo_column_list1.append(item1)
    
    #-----pair denovo columns (denovo_seq and aa_score) into list of tuples
    column_name_tuple_list1 = []
    
    for position1,item1 in enumerate(denovo_column_list1):
        for item2 in range((position1+1), len(denovo_column_list1)):
            if 'denovo_seq_' in item1:
                item3 = item1.replace('denovo_seq_', '')
                if item3 in denovo_column_list1[item2]:
                    column_name_tuple_list1.append((item1, denovo_column_list1[item2]))
    
    #-----create new columns in copied dataframe by the addition of denovo_seq + aa_score columns using '$' character to denote string fusion point
    new_columns_list1 = []
    
    for item1 in column_name_tuple_list1:
        temp_column_name1 = item1[0].replace('denovo_seq_', '') + '_masked'
        temp_df1[temp_column_name1] = temp_df1[item1[0]].astype(str) + '$' + temp_df1[item1[1]].astype(str)
        new_columns_list1.append(temp_column_name1)
    
    #-----iterate through dataframe and remove PTMs (defined in 'PTM_list1' input list), then split strings on '$' character, 
    #-----then turn amino acid residues into 'X' characters when score is below 'score_cutoff' input value
    temp_new_col1 = []
    #list of tuples; tuples contain new col name, new col contents
    temp_colname_colcontents_list1 = []
    
    for row1 in temp_df1.items():
        if row1[0] in new_columns_list1:
            for item1 in row1[1]:
                item1_copy = item1
                #---remove PTMs
                for item2 in PTM_list1:
                    item1_copy = item1_copy.replace(item2, '')
                #---split string, turn into list with 2 entries: [0]-sequence, [1]-aa_score
                item1_copy = item1_copy.split('$')
                temp_seq1 = item1_copy[0]
                temp_score1 = item1_copy[1]
                #---reformat and split score into list (likely list of str)
                temp_score1 = temp_score1.replace('[', '').replace(']', '').replace(' ', '')
                temp_score1 = temp_score1.split(',')
                #---iterate through temp_score1 list and compare to score_cutoff input
                #---if < score_cutoff, change corresponding position in temp_seq1 into 'X'
                for position1,temp_aa_score1 in enumerate(temp_score1):
                    if int(temp_aa_score1) < int(score_cutoff):
                        temp_seq1 = list(temp_seq1)
                        #---note that there are casanovo sequences where score chain lengths
                        #---are mismatched with sequence lengths; try/except attempt to ignore these
                        try:
                            temp_seq1[position1] = 'X'
                        except:
                            pass
                        temp_seq1 = ''.join(temp_seq1)
                #---add temp_seq1 sequence to list
                temp_new_col1.append(temp_seq1)
            temp_colname_colcontents_list1.append((str(row1[0]), temp_new_col1))
            temp_new_col1 = []
    
    #-----add new columns to dataframe then return dataframe
    for item1 in temp_colname_colcontents_list1:
        temp_df1[item1[0]] = item1[1]
    
    return temp_df1

def create_BLAST_query_file(input_dataframe1, prefix_suffix_list1, blast_filename_1):
    #-----input: dataframe containing denovo_seq reformatted columns and/or denovo_seq masked columns, list of column prefixes/suffixes associated with raw file conversion
    #-----input: output filename ('blast_filename_1')
    #-----output: fasta file for blast where fasta documents are labeled according to "scan.spectrum_ID.column_name"
    
    #-----select columns for adding to fasta file
    column_list1 = []
    for item1 in input_dataframe1.columns:
        for item2 in prefix_suffix_list1:
            if ((str(item2)) in item1) and (('_masked' in item1) or ('_reformatted' in item1)):
                column_list1.append(item1)
    
    #-----create temporary data objects
    temp_concat_sequences = pd.Series([])
    temp_all_concat_sequences = pd.Series([])
    
    for item1 in input_dataframe1.columns:
        if item1 in column_list1:
            temp_concat_sequences = input_dataframe1['scan'].astype(str) + '.' + input_dataframe1['spectrum_ID'].astype(str) + '.' + str(item1) + '#' + input_dataframe1[item1]
        temp_all_concat_sequences = pd.concat([temp_all_concat_sequences, temp_concat_sequences])
    
    #-----write fasta documents to file
    with open(blast_filename_1, 'w') as f:
        for item1 in temp_all_concat_sequences:
            temp_doc1 = item1.split('#')
            temp_line1 = '>' + str(temp_doc1[0]) + '\n'
            temp_line2 = str(temp_doc1[1]) + '\n'
            f.write(temp_line1)
            f.write(temp_line2)

def reduce_Blast_file(input_file1, output_file1):
    #-----input: output fasta file from create_BLAST_query_file function
    #-----output: condensed fasta file for BLAST search, dictionary for decoding blast results
    
    #------open file
    with open(input_file1, 'r') as f:
        file1 = f.readlines()
    
    #-----create peptide-description dictionary
    pep_description_dict1 = {}
    temp_description = ''
    temp_pep = ''
    
    for line1 in file1:
        #---collect fasta entries
        if line1[0] == '>':
            temp_description = line1.replace('>', '').replace('\n', '')
        else:
            temp_pep = line1.replace('\n', '')
            #---add fasta entry to dictionary
            try:
                pep_description_dict1[temp_pep] += [temp_description]
            except:
                pep_description_dict1[temp_pep] = [temp_description]
            #---clear variables
            temp_description = ''
            temp_pep = ''
    
    #-----create new BLAST fasta file
    temp_count1 = 0
    seqnum_pep_dict1 = {}
    
    with open(output_file1, 'w') as g:
        for item1 in pep_description_dict1.keys():
            temp_line1 = '>' + 'seq_' + str(temp_count1) + '\n'
            temp_line2 = str(item1) + '\n'
            g.write(temp_line1)
            g.write(temp_line2)
            temp_line3 = 'seq_' + str(temp_count1)
            seqnum_pep_dict1[temp_line3] = str(item1)
            temp_count1 += 1
        
    #-----return
    return pep_description_dict1, seqnum_pep_dict1

def associate_evalues(main_df1, blast_df1, col_name1, G_P_suffix):
    #-----input: combined dataframe containing database and denovo values, blast result dataframe containing peptide information, name of combined dataframe column containing peptides used for BLAST
    #-----output: combined dataframe with new column for evalues corresponding to BLASTed peptide results
    
    #-----create peptide-evalue dictionary
    temp_dict1 = {}
    for row1 in blast_df1.itertuples():
        temp_dict1[row1.sequence] = row1.evalue
    
    #-----iterate through main dataframe and assign evalues to peptides
    new_col_list1 = []
    
    pep_col1 = main_df1[col_name1]
    for item1 in pep_col1:
        try:
            new_col_list1.append(temp_dict1[item1])
        except:
            new_col_list1.append(-1)
    
    new_col_name1 = str(col_name1) + '_evalues_' + str(G_P_suffix)
    mini_df2 = pd.DataFrame([pd.Series(pep_col1), pd.Series(new_col_list1)]).transpose()
    mini_df2.columns = (col_name1, new_col_name1)
    
    main_df2 = main_df1.copy()
    main_df2[new_col_name1] = mini_df2[new_col_name1]

    return main_df2

def extract_Gene_Sequence(genome_fasta1, contig_name, start_position0, stop_position0):
    #-----input: location of genome in fasta format, contig name, and start and stop positions of denovo peptides
    #-----output: returns gene sequence, new start position and new stop position; gene is bookended by a start codon and stop codon in the same frame as the input query
    
    #-----read genome file
    with open(genome_fasta1, 'r') as f:
        file1 = f.readlines()
    
    #-----iterate through genome file until appropriate contig is found, then store contig sequence
    contig_sequence1 = ''
    start1 = False
    
    for line1 in file1:
        if line1[0] == '>':
            if start1 == True:
                start1 = False
                break
            if line1.replace('>', '').replace('\n', '') == contig_name:
                start1 = True
        elif start1 == True:
            contig_sequence1 += line1.replace('\n', '')
    
    #-----fix sequence oriention if necessary (is target sequence forward or reverse relative to contig?)
    reverse_input = False
    if stop_position0 < start_position0:
        temp_complement = ''
        for base1 in contig_sequence1:
            if base1 == 'A':
                temp_complement += 'T'
            elif base1 == 'T':
                temp_complement += 'A'
            elif base1 == 'C':
                temp_complement += 'G'
            elif base1 == 'G':
                temp_complement += 'C'
            else:
                temp_complement += 'X'
        temp_reverse_complement = temp_complement[::-1]
        contig_sequence1 = temp_reverse_complement
        start_position1 = len(contig_sequence1) - start_position0
        stop_position1 = len(contig_sequence1) - stop_position0
        reverse_input = True
    else:
        start_position1 = start_position0 - 1
        stop_position1 = stop_position0 - 1

    #-----iterate through contig 3 nucleotides at a time both upstream and downstream from target sequence; stop at start codon upstream; stop at stop codon downstream
    start_codon_list1 = ['ATG', 'GTG', 'TTG']
    stop_codon_list1 = ['TAG', 'TAA', 'TGA']
    contig_sequence2_whole = contig_sequence1[start_position1:(stop_position1+1)]
    contig_sequence2_5prime = contig_sequence1[:start_position1]
    contig_sequence2_3prime = contig_sequence1[(stop_position1+1):]
    start_expansion_count = 0
    stop_expansion_count = 0
    temp_triplet = ''
    
    #---find stop codon or end of contig, whichever comes first
    temp_triplet = ''
    for residue1 in contig_sequence2_3prime:
        temp_triplet += residue1
        stop_expansion_count += 1
        if len(temp_triplet) >= 3:
            contig_sequence2_whole += temp_triplet
            if temp_triplet in stop_codon_list1:
                break
            temp_triplet = ''
    
    #---find start codon or start of contig, whichever comes first
    temp_triplet = ''
    for residue1 in contig_sequence2_5prime[::-1]:
        temp_triplet += residue1
        start_expansion_count += 1
        if len(temp_triplet) >= 3:
            contig_sequence2_whole = temp_triplet[::-1] + contig_sequence2_whole
            if temp_triplet[::-1] in start_codon_list1:
                break
            temp_triplet = ''
    
    #-----calculate whole-gene start and stop parameters in same format as input start and stop parameters
    if reverse_input == False:
        start_position2 = start_position1 - start_expansion_count
        stop_position2 = stop_position1 + stop_expansion_count
    else:
        start_position2 = len(contig_sequence1) - start_position1 + start_expansion_count
        stop_position2 = len(contig_sequence1) - stop_position1 - stop_expansion_count
    
    return contig_sequence2_whole, start_position2, stop_position2

def create_Denovo_Protein_Fasta(denovo_df1, col_name1, accession_prefix1, output_file_name1):
    #-----input: dataframe containing DNA sequences in a column, name of column containing DNA sequences, accession prefixes for fasta labels, output file name
    #-----output: protein fasta file
    
    #-----select dataframe column containing DNA sequences
    dna_seq_list1 = denovo_df1[col_name1]
    
    #-----iterate through column
    prot_seq_list1 = []
    
    for item1 in dna_seq_list1:
        prot_seq_list1.append(translate(item1).replace('*', ''))
    
    #-----remove duplicate protein sequences
    prot_seq_list1 = list(set(prot_seq_list1))
    
    #-----output protein sequences into fasta file
    with open(output_file_name1, 'w') as f:
        temp_count = 0
        for item1 in prot_seq_list1:
            temp_line1 = '>' + str(accession_prefix1) + '_' + str(temp_count) + '\n'
            temp_line2 = str(item1) + '\n'
            f.write(temp_line1)
            f.write(temp_line2)
            temp_count += 1


