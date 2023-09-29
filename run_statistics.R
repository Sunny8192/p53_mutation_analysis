# load data
df = read.csv("/home/qpan/Desktop/p_P53/manuscript/features_p53.csv")

# select the DNA-binding domain
df = df[which((df$pos <= 293) & (df$pos >= 93)),]

# select the computational biophysical measurements
features = c("plddt", "rsa", "phi", "psi", "residue_depth", "na_distance", "AB_ppi_distance", 
             "BC_ppi_distance", "BD_ppi_distance", "zn_distance", "delta_vibrational_entropy", 
             "dynamut1_ddG", "mcsm_ddG", "duet_ddG", "sdm_ddG", "encom_ddG", "dynamut2_ddG", 
             "na1_ddG", "na2_ddG", "AB_ppi1_ddG", "BC_ppi1_ddG", "BD_ppi1_ddG", "AB_ppi2_ddG", 
             "BC_ppi2_ddG", "BD_ppi2_ddG", "saafec_ddG")

for (feat in features) {
    p_value = wilcox.test(df[, feat] ~ label, data=df)$p.value
    print(feat)
    print(p_value)
}


