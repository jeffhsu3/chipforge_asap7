import gdstk

# ASAP7 Layers
L_WELL = (1, 0)
L_FIN = (2, 0)
L_GATE = (7, 0)
L_DUMMY = (8, 0)
L_GCUT = (10, 0)
L_ACTIVE = (11, 0)
L_NSELECT = (12, 0)
L_PSELECT = (13, 0)
L_LIG = (16, 0)
L_LISD = (17, 0)
L_V0 = (18, 0)
L_M1 = (19, 0)
L_M2 = (20, 0)
L_V1 = (21, 0)
L_V2 = (25, 0)
L_M3 = (30, 0)
L_BOUNDARY = (100, 0)
L_TEXT = (101, 0)

# Pin Layers
P_M1 = (19, 251)
P_M2 = (20, 251)
P_M3 = (30, 251)

lib = gdstk.Library()
cell = lib.new_cell("sram_cell_pushrule_6t")

# Dimensions
width = 0.216
height = 0.166

# 1. Boundary
boundary = gdstk.rectangle((0, 0), (width, height), layer=L_BOUNDARY[0], datatype=L_BOUNDARY[1])
cell.add(boundary)

# 2. Fins (Horizontal)
fin_w = 0.007
fin_y_bot = 0.030
fin_y_mid = 0.083
fin_y_top = 0.136

cell.add(gdstk.rectangle((0, fin_y_bot - fin_w/2), (width, fin_y_bot + fin_w/2), layer=L_FIN[0], datatype=L_FIN[1]))
cell.add(gdstk.rectangle((0, fin_y_mid - fin_w/2), (width, fin_y_mid + fin_w/2), layer=L_FIN[0], datatype=L_FIN[1]))
cell.add(gdstk.rectangle((0, fin_y_top - fin_w/2), (width, fin_y_top + fin_w/2), layer=L_FIN[0], datatype=L_FIN[1]))

# 3. Active and Select
cell.add(gdstk.rectangle((0, 0.015), (width, 0.045), layer=L_ACTIVE[0], datatype=L_ACTIVE[1]))
cell.add(gdstk.rectangle((0, 0.065), (width, 0.100), layer=L_ACTIVE[0], datatype=L_ACTIVE[1]))
cell.add(gdstk.rectangle((0, 0.120), (width, 0.150), layer=L_ACTIVE[0], datatype=L_ACTIVE[1]))

cell.add(gdstk.rectangle((0, 0.010), (width, 0.050), layer=L_NSELECT[0], datatype=L_NSELECT[1]))
cell.add(gdstk.rectangle((0, 0.060), (width, 0.105), layer=L_PSELECT[0], datatype=L_PSELECT[1]))
cell.add(gdstk.rectangle((0, 0.115), (width, 0.155), layer=L_NSELECT[0], datatype=L_NSELECT[1]))
cell.add(gdstk.rectangle((0, 0.060), (width, 0.105), layer=L_WELL[0], datatype=L_WELL[1])) # NWELL for PMOS

# 4. Gates (Vertical)
gate_w = 0.020
cpp = 0.054
g1_x = cpp
g2_x = 3 * cpp

cell.add(gdstk.rectangle((g1_x - gate_w/2, 0.0), (g1_x + gate_w/2, height), layer=L_GATE[0], datatype=L_GATE[1]))
cell.add(gdstk.rectangle((g2_x - gate_w/2, 0.0), (g2_x + gate_w/2, height), layer=L_GATE[0], datatype=L_GATE[1]))

# 5. Local Interconnects (LISD for Source/Drain, LIG for Gate)
lisd_w = 0.024
cell.add(gdstk.rectangle((0.027 - lisd_w/2, 0.015), (0.027 + lisd_w/2, 0.045), layer=L_LISD[0], datatype=L_LISD[1]))
cell.add(gdstk.rectangle((0.189 - lisd_w/2, 0.120), (0.189 + lisd_w/2, 0.150), layer=L_LISD[0], datatype=L_LISD[1]))
cell.add(gdstk.rectangle((0.027 - lisd_w/2, 0.065), (0.027 + lisd_w/2, 0.100), layer=L_LISD[0], datatype=L_LISD[1]))
cell.add(gdstk.rectangle((0.189 - lisd_w/2, 0.065), (0.189 + lisd_w/2, 0.100), layer=L_LISD[0], datatype=L_LISD[1]))

# 6. M1 (Horizontal)
m1_w = 0.018
vss_bot_y = 0.015
vdd_y = 0.083
vss_top_y = 0.151

cell.add(gdstk.rectangle((0, vss_bot_y - m1_w/2), (width, vss_bot_y + m1_w/2), layer=L_M1[0], datatype=L_M1[1]))
cell.add(gdstk.Label("VSS", (0.108, vss_bot_y), layer=P_M1[0], texttype=P_M1[1]))

cell.add(gdstk.rectangle((0, vdd_y - m1_w/2), (width, vdd_y + m1_w/2), layer=L_M1[0], datatype=L_M1[1]))
cell.add(gdstk.Label("VDD", (0.108, vdd_y), layer=P_M1[0], texttype=P_M1[1]))

cell.add(gdstk.rectangle((0, vss_top_y - m1_w/2), (width, vss_top_y + m1_w/2), layer=L_M1[0], datatype=L_M1[1]))
cell.add(gdstk.Label("VSS", (0.108, vss_top_y), layer=P_M1[0], texttype=P_M1[1]))

# 7. M2 (Vertical) - BL, BLN
m2_w = 0.018
bl_x = 0.054
bln_x = 0.162

cell.add(gdstk.rectangle((bl_x - m2_w/2, 0), (bl_x + m2_w/2, height), layer=L_M2[0], datatype=L_M2[1]))
cell.add(gdstk.Label("BL", (bl_x, 0.083), layer=P_M2[0], texttype=P_M2[1]))

cell.add(gdstk.rectangle((bln_x - m2_w/2, 0), (bln_x + m2_w/2, height), layer=L_M2[0], datatype=L_M2[1]))
cell.add(gdstk.Label("BLN", (bln_x, 0.083), layer=P_M2[0], texttype=P_M2[1]))

# 8. M3 (Horizontal) - WL
m3_w = 0.018
wl_y = 0.050

cell.add(gdstk.rectangle((0, wl_y - m3_w/2), (width, wl_y + m3_w/2), layer=L_M3[0], datatype=L_M3[1]))
cell.add(gdstk.Label("WL", (0.108, wl_y), layer=P_M3[0], texttype=P_M3[1]))

lib.write_gds("tech/gds/sram_pushrule_6t.gds")
print("Successfully generated tech/gds/sram_pushrule_6t.gds")
