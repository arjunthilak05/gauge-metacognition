"""
GAUGE: Grading Awareness of Uncertainty and Grounded Epistemics
================================================================
Kaggle Benchmark Task — Clean version for "Add Models" execution.

This file contains ONLY the task definition and evaluation data.
No analytics, no visualizations, no external dependencies beyond kbench.

For full analysis (metrics, figures, CSV export), see metacog_full.py.
"""

import kaggle_benchmarks as kbench
import pandas as pd
import re
from dataclasses import dataclass


# ============================================================
#  ITEMS (270: 122 math + 115 logic + 33 factual, CTT-selected)
# ============================================================

ITEMS = [
    ('David earned money over 2 days. On Monday, David earned $119. On Tuesday, David earned $300. David was wearing a red hat that day. How much did David earn in total?', '419', 3, 'math'),
    ('Tom wants to buy a bicycle that costs $2844. The store is offering a 50% discount today. The store\'s loyalty program showed Tom had 40 points, which weren\'t redeemable yet. The cashier mentioned they had sold 36 items earlier that day to someone else. How much does Tom pay after the discount?', '1422', 6, 'math'),
    ('Maria has 39 bananas. Maria gives 17 to Rachel and buys 42 more. How many bananas does Maria have now?', '64', 2, 'math'),
    ('Sarah has 199 pears. The store had been open since 1987. Sarah gives 75 to Aisha and buys 113 more. How many pears does Sarah have now?', '237', 3, 'math'),
    ('Carlos earned money over 3 days. On Monday, Carlos earned $294. On Tuesday, Carlos earned $513. On Wednesday, Carlos earned $189. The cashier mentioned they had sold 2 items earlier that day to someone else. How much did Carlos earn in total?', '996', 4, 'math'),
    ('Tom has 20 oranges. Tom was wearing a red hat that day. Tom gives 8 to Sarah and buys 136 more. How many oranges does Tom have now?', '148', 3, 'math'),
    ('Rachel goes to a store to buy notebooks. Each notebook costs $1. Rachel buys 17 of them. How much does Rachel spend in total?', '17', 1, 'math'),
    ('Aisha earned a total of $1280 over three days. There were 6 items in the clearance bin, but Aisha wasn\'t interested. On the first day, Aisha earned $580. On the second day, Aisha earned $402. How much did Aisha earn on the third day?', '298', 4, 'math'),
    ('Rachel bought 10 markers and paid $1690 in total. Each marker cost the same amount. The store\'s loyalty program showed Rachel had 21 points, which weren\'t redeemable yet. How much did each marker cost?', '169', 5, 'math'),
    ('Nadia has 661 bananas and 247 apples. Nadia gives 178 bananas to a friend. Nadia noticed that 27 other customers were in the store. How many bananas does Nadia have now?', '483', 4, 'math'),
    ('Sarah has $31. Sarah spends $7 on a book. Sarah spends $10 on a gift. How much money does Sarah have left?', '14', 2, 'math'),
    ('Sarah goes shopping with $938. Sarah buys 5 markers at $79 each. There were 29 items in the clearance bin, but Sarah wasn\'t interested. Sarah buys 1 pens at $48 each. Sarah buys 4 notebooks at $56 each. How much money does Sarah have left?', '271', 4, 'math'),
    ('Kenji has 118 cookies. Kenji\'s favorite color is blue. Lin has 3 times as many cookies as Kenji. How many cookies do they have together?', '472', 3, 'math'),
    ('Aisha has $6163. Aisha spends $1799 on a gift. Aisha spends $228 on a book. Aisha had 48 coupons in a drawer at home but forgot to bring them. Aisha spends $2314 on coffee. How much money does Aisha have left?', '1822', 5, 'math'),
    ('Priya has 416 stickers and wants to share them equally among 4 friends. Priya was wearing a red hat that day. How many stickers does each friend get?', '104', 3, 'math'),
    ('David has 28 mangoes. David gives 11 to Maria and buys 27 more. How many mangoes does David have now?', '44', 1, 'math'),
    ('James has 12 cookies and wants to share them equally among 6 friends. How many cookies does each friend get?', '2', 1, 'math'),
    ('James has 1566 cookies. The store also had 25 envelopes on display, but James didn\'t buy any. Nadia has 2 times as many cookies as James. James\'s friend had mentioned wanting 45 cookies, but James didn\'t get any for them. How many cookies do they have together?', '4698', 6, 'math'),
    ('Nadia builds 38 widgets per day. The store had been open since 1987. If Nadia works for 8 days, how many widgets does Nadia produce?', '304', 3, 'math'),
    ('Omar has 548 bananas and 417 mangoes. Omar gives 200 bananas to a friend. Omar\'s friend had mentioned wanting 46 bananas, but Omar didn\'t get any for them. How many bananas does Omar have now?', '348', 4, 'math'),
    ('Kenji saves $109 every week for 5 weeks. During that time, Kenji spends $5 on concert tickets. Kenji\'s friend had mentioned wanting 39 items, but Kenji didn\'t get any for them. How much money does Kenji have at the end?', '540', 4, 'math'),
    ('Tom earned money over 3 days. On Monday, Tom earned $701. On Tuesday, Tom earned $4744. On Wednesday, Tom earned $5221. The shelf had 37 empty spots where items used to be. How much did Tom earn in total?', '10666', 5, 'math'),
    ('Lin bought 3 markers and paid $441 in total. On the way there, Lin passed 29 houses. Each marker cost the same amount. How much did each marker cost?', '147', 4, 'math'),
    ('Fatima goes shopping with $4509. Fatima buys 8 erasers at $156 each. Fatima had 46 coupons in a drawer at home but forgot to bring them. Fatima buys 5 markers at $113 each. Fatima buys 6 folders at $65 each. How much money does Fatima have left?', '2306', 5, 'math'),
    ('Alex earned a total of $15117 over three days. On the way there, Alex passed 10 houses. The store also had 29 batteries on display, but Alex didn\'t buy any. On the first day, Alex earned $2220. On the second day, Alex earned $6424. How much did Alex earn on the third day?', '6473', 6, 'math'),
    ('Kenji bought 9 notebooks and paid $315 in total. The shelf had 35 empty spots where notebooks used to be. Each notebook cost the same amount. How much did each notebook cost?', '35', 4, 'math'),
    ('Priya has 188 bananas and 180 oranges. There were 38 items in the clearance bin, but Priya wasn\'t interested. Priya gives 77 bananas to a friend. How many bananas does Priya have now?', '111', 4, 'math'),
    ('Aisha wants to buy a camera that costs $748. The store is offering a 50% discount today. Aisha noticed that 49 other customers were in the store. How much does Aisha pay after the discount?', '374', 4, 'math'),
    ('Nadia earned money over 4 days. On Monday, Nadia earned $17. On Tuesday, Nadia earned $18. On Wednesday, Nadia earned $26. On Thursday, Nadia earned $16. How much did Nadia earn in total?', '77', 2, 'math'),
    ('Tom has 340 mangoes. The receipt was printed on recycled paper. Tom gives 131 to Aisha and buys 472 more. How many mangoes does Tom have now?', '681', 3, 'math'),
    ('Finn is a Snazzles. Every Snazzles is a Murnips. Every Murnips is a Flimbers. All Blorps are wise. Every Flimbers is a Wumpuses. Is Finn a Quibbles?', 'Cannot be determined', 4, 'logic'),
    ('Mina\'s favorite color is orange. Jake enjoys gardening. Finn\'s favorite color is yellow. Gina\'s favorite color is blue. Iris does not like blue or orange. Iris\'s favorite color is red. Noel does not like blue. What is Noel\'s favorite color?', 'purple', 3, 'logic'),
    ('If something is a Snazzles, it might be wise. Ora is a Quibbles. Every Quibbles is a Tazzles. Every Tazzles is a Grumpkins. Every Grumpkins is a Fizzgigs. Is Ora a Fizzgigs?', 'Yes', 3, 'logic'),
    ('All Snazzles are kind. Ora is a Murnips. All Murnips are quiet. All quiet things are kind. Is Ora not kind?', 'No', 4, 'logic'),
    ('Hugo is a Drazzles. All Drazzles are fast. If something is fast, then it is wise. Some Quibbles are tall. If something is wise, then it is kind. Is Hugo clever?', 'Cannot be determined', 5, 'logic'),
    ('All Vexlings are wise. Kira is a Drazzles. All Drazzles are quiet. If something is quiet, then it is wise. Is Kira wise?', 'Yes', 3, 'logic'),
    ('Paul is a Vexlings. Some Tazzles are clever. All Vexlings are kind. If something is kind, then it is friendly. Is Paul friendly?', 'Yes', 4, 'logic'),
    ('Paul is a Murnips. All Murnips are wise. If something is wise, then it is fast. If something is fast, then it is clever. All Fizzgigs are wise. Some Drogons are happy. Is Paul clever?', 'Yes', 6, 'logic'),
    ('Cora is a Vexlings. All Vexlings are brave. Some Flimbers are quiet. If something is brave, then it is fast. Is Cora wise?', 'Cannot be determined', 3, 'logic'),
    ('Paul is a Drogons. Some Plonkers are friendly. All Drogons are fast. If something is fast, then it is happy. Is Paul happy?', 'Yes', 3, 'logic'),
    ('Hugo\'s favorite color is red. Cora enjoys gardening. Finn\'s favorite color is blue. Ora does not like blue or purple. Ora\'s favorite color is orange. Paul does not like orange. What is Paul\'s favorite color?', 'purple', 5, 'logic'),
    ('Gina\'s favorite color is purple. Eve does not like blue or purple. Eve\'s favorite color is green. Iris does not like blue or purple. Noel enjoys painting. Iris\'s favorite color is orange. Paul does not like purple. What is Paul\'s favorite color?', 'blue', 3, 'logic'),
    ('Dan is a Murnips. Every Murnips is a Wumpuses. Every Wumpuses is a Snazzles. Is Dan a Wumpuses?', 'Yes', 2, 'logic'),
    ('Leo is a Wumpuses. All Grumpkins are kind. Every Wumpuses is a Zephlins. Every Zephlins is a Snazzles. Every Snazzles is a Drazzles. Is Leo a Drogons?', 'Cannot be determined', 4, 'logic'),
    ('Leo\'s favorite color is blue. Cora does not like blue or red. Eve enjoys cooking. Gina enjoys painting. Cora\'s favorite color is yellow. Ava does not like purple or blue. Ava\'s favorite color is green. Jake\'s favorite color is red. Paul does not like yellow. What is Paul\'s favorite color?', 'purple', 6, 'logic'),
    ('Mina is a Fizzgigs. All Fizzgigs are tall. If something is tall, then it is clever. Is Mina clever?', 'Yes', 2, 'logic'),
    ('Some Plonkers are fast. Paul is a Grumpkins. Every Grumpkins is a Drazzles. Every Drazzles is a Kelpoids. Every Kelpoids is a Blorps. Every Blorps is a Wumpuses. Is Paul a Flimbers?', 'Cannot be determined', 5, 'logic'),
    ('Paul is a Vexlings. All Flimbers are fast. All Vexlings are kind. If something is kind, then it is wise. If something is wise, then it is fast. Some Drogons are strong. Is Paul fast?', 'Yes', 6, 'logic'),
    ('Mina is a Plonkers. All Plonkers are quiet. All quiet things are clever. If something is a Tazzles, it might be happy. Is Mina not clever?', 'No', 4, 'logic'),
    ('Some Drogons are brave. Cora is a Grumpkins. All Grumpkins are tall. If something is tall, then it is fast. Is Cora fast?', 'Yes', 3, 'logic'),
    ('Eve\'s favorite color is yellow. Hugo\'s favorite color is red. Ora enjoys chess. Leo\'s favorite color is blue. Kira does not like yellow. What is Kira\'s favorite color?', 'green', 4, 'logic'),
    ('Dan\'s favorite color is blue. Mina enjoys painting. Eve enjoys gardening. Ora does not like purple or green. Ora\'s favorite color is red. Iris does not like red or green. Iris\'s favorite color is yellow. Noel\'s favorite color is green. Finn does not like blue. What is Finn\'s favorite color?', 'purple', 6, 'logic'),
    ('Noel\'s favorite color is red. Finn\'s favorite color is purple. Leo does not like green or red. Hugo enjoys cooking. Leo\'s favorite color is orange. Dan does not like purple. What is Dan\'s favorite color?', 'green', 3, 'logic'),
    ('Some Drazzles are wise. Hugo is a Plonkers. All Plonkers are friendly. If something is friendly, then it is clever. Is Hugo strong?', 'Cannot be determined', 3, 'logic'),
    ('Finn is a Wumpuses. Some Drazzles are kind. Every Wumpuses is a Plonkers. Every Plonkers is a Zephlins. Every Zephlins is a Murnips. Is Finn a Grumpkins?', 'Cannot be determined', 3, 'logic'),
    ('Noel is a Zephlins. All Zephlins are strong. All Drazzles are clever. If something is strong, then it is quiet. Is Noel wise?', 'Cannot be determined', 4, 'logic'),
    ('If something is a Drazzles, it might be friendly. Kira is a Flimbers. All Flimbers are quiet. If something is quiet, then it is strong. Is Kira strong?', 'Yes', 3, 'logic'),
    ('Ava\'s favorite color is purple. Finn does not like purple or green. Finn\'s favorite color is yellow. Jake\'s favorite color is red. Kira does not like yellow or red. Kira\'s favorite color is green. Gina enjoys painting. Iris does not like red. What is Iris\'s favorite color?', 'blue', 5, 'logic'),
    ('Eve\'s favorite color is orange. Kira\'s favorite color is yellow. Ora\'s favorite color is purple. Cora does not like yellow or purple. Mina enjoys cooking. Cora\'s favorite color is blue. Noel does not like blue. Ava enjoys painting. What is Noel\'s favorite color?', 'red', 6, 'logic'),
    ('Paul is a Grumpkins. All Flimbers are brave. All Grumpkins are quiet. All quiet things are wise. Is Paul wise?', 'Yes', 4, 'logic'),
    ('What is the capital of France?', 'Paris', 1, 'factual'),
    ('What gas do plants absorb during photosynthesis?', 'Carbon dioxide', 1, 'factual'),
    ('What is the largest continent by area?', 'Asia', 1, 'factual'),
    ('How many planets are in our solar system?', '8', 1, 'factual'),
    ('In what year did World War II end?', '1945', 1, 'factual'),
    ('What country has the largest population?', 'India', 1, 'factual'),
    ('In what year did humans first land on the Moon?', '1969', 1, 'factual'),
    ('Who was the first president of the United States?', 'George Washington', 1, 'factual'),
    ('Carlos has 42 cookies and wants to share them equally among 7 friends. How many cookies does each friend get?', '6', 1, 'math'),
    ('Fatima goes to a store to buy rulers. Each ruler costs $28. Fatima buys 18 of them. How much does Fatima spend in total?', '504', 1, 'math'),
    ('Marcus has 28 books and wants to share them equally among 7 friends. How many books does each friend get?', '4', 1, 'math'),
    ('Rachel goes to a store to buy markers. Each marker costs $50. Rachel buys 19 of them. How much does Rachel spend in total?', '950', 1, 'math'),
    ('Priya has 21 mangoes. Priya gives 5 to Maria and buys 2 more. How many mangoes does Priya have now?', '18', 1, 'math'),
    ('Fatima goes to a store to buy erasers. Each eraser costs $1. Fatima buys 20 of them. How much does Fatima spend in total?', '20', 1, 'math'),
    ('Nadia goes to a store to buy markers. Each marker costs $50. Nadia buys 4 of them. How much does Nadia spend in total?', '200', 1, 'math'),
    ('Nadia has 45 pencils and wants to share them equally among 5 friends. How many pencils does each friend get?', '9', 1, 'math'),
    ('Tom has 43 oranges. Tom gives 13 to Priya and buys 11 more. How many oranges does Tom have now?', '41', 1, 'math'),
    ('David has 2 pears. David gives 1 to Tom and buys 26 more. How many pears does David have now?', '27', 1, 'math'),
    ('Omar has 33 books and wants to share them equally among 3 friends. How many books does each friend get?', '11', 1, 'math'),
    ('Nadia goes to a store to buy markers. Each marker costs $38. Nadia buys 2 of them. How much does Nadia spend in total?', '76', 1, 'math'),
    ('James has 36 marbles and wants to share them equally among 3 friends. How many marbles does each friend get?', '12', 1, 'math'),
    ('Fatima goes to a store to buy notebooks. Each notebook costs $20. Fatima buys 20 of them. How much does Fatima spend in total?', '400', 1, 'math'),
    ('Rachel has 36 stickers and wants to share them equally among 4 friends. How many stickers does each friend get?', '9', 1, 'math'),
    ('Sarah has 16 peaches. Sarah gives 5 to Alex and buys 3 more. How many peaches does Sarah have now?', '14', 1, 'math'),
    ('Omar goes to a store to buy erasers. Each eraser costs $7. Omar buys 9 of them. How much does Omar spend in total?', '63', 1, 'math'),
    ('Nadia has 27 stickers and wants to share them equally among 3 friends. How many stickers does each friend get?', '9', 1, 'math'),
    ('Leo is a Drogons. Every Drogons is a Grumpkins. Is Leo a Kelpoids?', 'Cannot be determined', 1, 'logic'),
    ('Mina is a Snazzles. Every Snazzles is a Fizzgigs. Is Mina a Tazzles?', 'Cannot be determined', 1, 'logic'),
    ('Noel is a Blorps. Every Blorps is a Drogons. Is Noel a Drazzles?', 'Cannot be determined', 1, 'logic'),
    ('Dan is a Wumpuses. Every Wumpuses is a Grumpkins. Is Dan a Grumpkins?', 'Yes', 1, 'logic'),
    ('Jake is a Plonkers. Every Plonkers is a Kelpoids. Is Jake a Kelpoids?', 'Yes', 1, 'logic'),
    ('Leo is a Drogons. Every Drogons is a Tazzles. Is Leo a Plonkers?', 'Cannot be determined', 1, 'logic'),
    ('Ben is a Snazzles. Every Snazzles is a Quibbles. Is Ben a Quibbles?', 'Yes', 1, 'logic'),
    ('Eve is a Drazzles. Every Drazzles is a Flimbers. Is Eve a Flimbers?', 'Yes', 1, 'logic'),
    ('Mina is a Snazzles. Every Snazzles is a Murnips. Is Mina a Murnips?', 'Yes', 1, 'logic'),
    ('Ora is a Murnips. Every Murnips is a Grumpkins. Is Ora a Grumpkins?', 'Yes', 1, 'logic'),
    ('Eve is a Murnips. Every Murnips is a Blorps. Is Eve a Blorps?', 'Yes', 1, 'logic'),
    ('Cora is a Blorps. All Blorps are fast. Is Cora fast?', 'Yes', 1, 'logic'),
    ('Mina is a Snazzles. Every Snazzles is a Grumpkins. Is Mina a Tazzles?', 'Cannot be determined', 1, 'logic'),
    ('Kira is a Drazzles. Every Drazzles is a Snazzles. Is Kira a Snazzles?', 'Yes', 1, 'logic'),
    ('Eve is a Vexlings. All Vexlings are fast. Is Eve fast?', 'Yes', 1, 'logic'),
    ('Leo is a Plonkers. Every Plonkers is a Wumpuses. Is Leo a Wumpuses?', 'Yes', 1, 'logic'),
    ('Who composed The Four Seasons?', 'Antonio Vivaldi', 2, 'factual'),
    ('What is the smallest country in the world by area?', 'Vatican City', 2, 'factual'),
    ('What is the chemical symbol for gold?', 'Au', 2, 'factual'),
    ('What desert is the largest hot desert in the world?', 'Sahara', 2, 'factual'),
    ('In what year did World War I begin?', '1914', 2, 'factual'),
    ('What is the capital of Canada?', 'Ottawa', 2, 'factual'),
    ('How many chromosomes do humans have?', '46', 2, 'factual'),
    ('What is the capital of Australia?', 'Canberra', 2, 'factual'),
    ('Aisha has $36. Aisha spends $2 on coffee. Aisha spends $11 on a book. How much money does Aisha have left?', '23', 2, 'math'),
    ('James has $71. James spends $15 on a shirt. James spends $9 on coffee. How much money does James have left?', '47', 2, 'math'),
    ('David goes to a store to buy folders. Each folder costs $49. David buys 5 of them. How much does David spend in total?', '245', 2, 'math'),
    ('Tom has 46 oranges. Tom gives 21 to Rachel and buys 42 more. How many oranges does Tom have now?', '67', 2, 'math'),
    ('David goes to a store to buy rulers. Each ruler costs $39. David buys 7 of them. How much does David spend in total?', '273', 2, 'math'),
    ('Lin has 28 cookies and wants to share them equally among 7 friends. How many cookies does each friend get?', '4', 2, 'math'),
    ('Priya goes to a store to buy pens. Each pen costs $16. Priya buys 14 of them. How much does Priya spend in total?', '224', 2, 'math'),
    ('Sarah has $24. Sarah spends $3 on a bus ticket. Sarah spends $1 on coffee. Sarah spends $9 on lunch. How much money does Sarah have left?', '11', 2, 'math'),
    ('Kenji has $43. Kenji spends $4 on a gift. Kenji spends $10 on a bus ticket. Kenji spends $10 on a shirt. How much money does Kenji have left?', '19', 2, 'math'),
    ('Kenji has 9 pencils and wants to share them equally among 3 friends. How many pencils does each friend get?', '3', 2, 'math'),
    ('Nadia earned money over 2 days. On Monday, Nadia earned $26. On Tuesday, Nadia earned $32. How much did Nadia earn in total?', '58', 2, 'math'),
    ('Priya goes to a store to buy erasers. Each eraser costs $28. Priya buys 19 of them. How much does Priya spend in total?', '532', 2, 'math'),
    ('Aisha goes to a store to buy pens. Each pen costs $42. Aisha buys 20 of them. How much does Aisha spend in total?', '840', 2, 'math'),
    ('Priya goes to a store to buy notebooks. Each notebook costs $35. Priya buys 12 of them. How much does Priya spend in total?', '420', 2, 'math'),
    ('James has 18 pears. James gives 6 to Carlos and buys 48 more. How many pears does James have now?', '60', 2, 'math'),
    ('Kenji has 33 stickers and wants to share them equally among 3 friends. How many stickers does each friend get?', '11', 2, 'math'),
    ('Nadia earned money over 4 days. On Monday, Nadia earned $34. On Tuesday, Nadia earned $29. On Wednesday, Nadia earned $36. On Thursday, Nadia earned $38. How much did Nadia earn in total?', '137', 2, 'math'),
    ('Lin has 5 pencils and wants to share them equally among 5 friends. How many pencils does each friend get?', '1', 2, 'math'),
    ('Cora is a Vexlings. All Vexlings are clever. If something is clever, then it is friendly. Is Cora fast?', 'Cannot be determined', 2, 'logic'),
    ('Dan is a Tazzles. All Tazzles are clever. If something is clever, then it is friendly. Is Dan friendly?', 'Yes', 2, 'logic'),
    ('Finn is a Snazzles. Every Snazzles is a Grumpkins. Is Finn a Grumpkins?', 'Yes', 2, 'logic'),
    ('Jake is a Blorps. All Blorps are fast. If something is fast, then it is brave. Is Jake clever?', 'Cannot be determined', 2, 'logic'),
    ('Ben is a Quibbles. Every Quibbles is a Vexlings. Every Vexlings is a Drogons. Is Ben a Drogons?', 'Yes', 2, 'logic'),
    ('Ora is a Drogons. Every Drogons is a Kelpoids. Every Kelpoids is a Zephlins. Is Ora a Zephlins?', 'Yes', 2, 'logic'),
    ('Dan is a Snazzles. Every Snazzles is a Tazzles. Every Tazzles is a Murnips. Is Dan a Vexlings?', 'Cannot be determined', 2, 'logic'),
    ('Ora is a Grumpkins. Every Grumpkins is a Snazzles. Is Ora a Snazzles?', 'Yes', 2, 'logic'),
    ('Iris is a Kelpoids. Every Kelpoids is a Fizzgigs. Every Fizzgigs is a Snazzles. Is Iris a Fizzgigs?', 'Yes', 2, 'logic'),
    ('Jake is a Grumpkins. All Grumpkins are wise. If something is wise, then it is tall. Is Jake happy?', 'Cannot be determined', 2, 'logic'),
    ('Ora is a Zephlins. All Zephlins are happy. If something is happy, then it is tall. Is Ora tall?', 'Yes', 2, 'logic'),
    ('Cora is a Murnips. All Murnips are friendly. If something is friendly, then it is wise. Is Cora wise?', 'Yes', 2, 'logic'),
    ('Leo is a Murnips. All Murnips are wise. If something is wise, then it is brave. Is Leo brave?', 'Yes', 2, 'logic'),
    ('Leo is a Murnips. All Murnips are quiet. Is Leo quiet?', 'Yes', 2, 'logic'),
    ('What is the highest mountain in Africa?', 'Mount Kilimanjaro', 3, 'factual'),
    ('Who composed the opera Carmen?', 'Georges Bizet', 3, 'factual'),
    ('What is the half-life of Carbon-14, approximately in years?', '5730', 3, 'factual'),
    ('What is the value of Avogadro\'s number, approximately?', '6.022e23', 3, 'factual'),
    ('What is the longest river in Europe?', 'Volga', 3, 'factual'),
    ('David earned money over 2 days. On Monday, David earned $457. On Tuesday, David earned $488. David\'s favorite color is blue. How much did David earn in total?', '945', 3, 'math'),
    ('Priya has 412 books and wants to share them equally among 4 friends. The store had been open since 1987. How many books does each friend get?', '103', 3, 'math'),
    ('Priya has 424 stickers and wants to share them equally among 8 friends. A song was playing on the radio in the background. How many stickers does each friend get?', '53', 3, 'math'),
    ('Alex has $655. Alex\'s favorite color is blue. Alex spends $92 on lunch. Alex spends $79 on a book. How much money does Alex have left?', '484', 3, 'math'),
    ('Lin wants to buy a guitar that costs $266. The store is offering a 50% discount today. Lin\'s favorite color is blue. How much does Lin pay after the discount?', '133', 3, 'math'),
    ('Rachel has 110 pencils. A song was playing on the radio in the background. Carlos has 3 times as many pencils as Rachel. How many pencils do they have together?', '440', 3, 'math'),
    ('Aisha earned money over 4 days. The store had been open since 1987. On Monday, Aisha earned $118. On Tuesday, Aisha earned $369. On Wednesday, Aisha earned $282. On Thursday, Aisha earned $87. How much did Aisha earn in total?', '856', 3, 'math'),
    ('Lin wants to buy a laptop that costs $128. The store is offering a 25% discount today. The receipt was printed on recycled paper. How much does Lin pay after the discount?', '96', 3, 'math'),
    ('Sarah has 38 mangoes. It was a sunny Wednesday afternoon. Sarah gives 3 to Alex and buys 310 more. How many mangoes does Sarah have now?', '345', 3, 'math'),
    ('Maria goes to a store to buy markers. Each marker costs $10. The receipt was printed on recycled paper. Maria buys 11 of them. How much does Maria spend in total?', '110', 3, 'math'),
    ('Marcus types 86 words per minute. Marcus\'s favorite color is blue. If Marcus works for 6 minutes, how many words does Marcus type?', '516', 3, 'math'),
    ('James goes to a store to buy rulers. Each ruler costs $21. James buys 3 of them. The receipt was printed on recycled paper. How much does James spend in total?', '63', 3, 'math'),
    ('Kira is a Tazzles. All Tazzles are wise. All wise things are happy. If something is a Vexlings, it might be tall. Is Kira brave?', 'Cannot be determined', 3, 'logic'),
    ('Some Murnips are clever. Kira is a Flimbers. All Flimbers are friendly. All friendly things are strong. Is Kira not strong?', 'No', 3, 'logic'),
    ('Ben is a Drazzles. All Drazzles are friendly. All Wumpuses are friendly. All friendly things are wise. Is Ben wise?', 'Yes', 3, 'logic'),
    ('Ben is a Vexlings. All Vexlings are happy. All happy things are brave. Some Wumpuses are happy. Is Ben brave?', 'Yes', 3, 'logic'),
    ('All Wumpuses are friendly. Eve is a Vexlings. All Vexlings are friendly. If something is friendly, then it is tall. Is Eve brave?', 'Cannot be determined', 3, 'logic'),
    ('Leo is a Vexlings. Every Vexlings is a Grumpkins. If something is a Fizzgigs, it might be clever. Every Grumpkins is a Blorps. Every Blorps is a Snazzles. Is Leo a Grumpkins?', 'Yes', 3, 'logic'),
    ('Eve is a Kelpoids. All Kelpoids are friendly. If something is a Tazzles, it might be brave. All friendly things are fast. Is Eve not fast?', 'No', 3, 'logic'),
    ('Cora enjoys hiking. Mina\'s favorite color is blue. Kira does not like purple or blue. Kira\'s favorite color is green. Hugo\'s favorite color is red. Ben does not like red. What is Ben\'s favorite color?', 'purple', 3, 'logic'),
    ('Gina is a Murnips. If something is a Grumpkins, it might be kind. All Murnips are wise. All wise things are kind. Is Gina not kind?', 'No', 3, 'logic'),
    ('Eve enjoys painting. Kira\'s favorite color is orange. Ben does not like yellow or orange. Ben\'s favorite color is purple. Paul\'s favorite color is yellow. Jake does not like purple. What is Jake\'s favorite color?', 'red', 3, 'logic'),
    ('What is the escape velocity from Earth\'s surface, approximately in km/s?', '11.2', 4, 'factual'),
    ('What was the capital of the Inca Empire?', 'Cusco', 4, 'factual'),
    ('In what year was the Treaty of Kuchuk Kainarji signed?', '1774', 4, 'factual'),
    ('What is the largest desert in the world?', 'Antarctic Desert', 4, 'factual'),
    ('What is the second-highest peak in the Karakoram range?', 'K2', 4, 'factual'),
    ('Kenji has $745. Kenji spends $131 on a gift. Kenji spends $103 on coffee. The store also had 7 magazines on display, but Kenji didn\'t buy any. Kenji spends $118 on a book. How much money does Kenji have left?', '393', 4, 'math'),
    ('Aisha wants to buy a guitar that costs $350. The store is offering a 10% discount today. Aisha noticed that 20 other customers were in the store. How much does Aisha pay after the discount?', '315', 4, 'math'),
    ('Aisha wants to buy a bicycle that costs $550. 24 of the items were slightly bruised, but Aisha kept them all. The store is offering a 10% discount today. How much does Aisha pay after the discount?', '495', 4, 'math'),
    ('Fatima saves $79 every week for 6 weeks. The store also had 35 candles on display, but Fatima didn\'t buy any. During that time, Fatima spends $47 on a birthday gift. How much money does Fatima have at the end?', '427', 4, 'math'),
    ('Rachel has $355. Rachel spends $21 on a book. Rachel spends $163 on coffee. There were 30 items in the clearance bin, but Rachel wasn\'t interested. Rachel spends $96 on a bus ticket. How much money does Rachel have left?', '75', 4, 'math'),
    ('Fatima builds 132 widgets per day. Fatima had 28 coupons in a drawer at home but forgot to bring them. If Fatima works for 2 days, how many widgets does Fatima produce?', '264', 4, 'math'),
    ('David saves $19 every week for 4 weeks. The shelf had 23 empty spots where items used to be. During that time, David spends $20 on a video game. How much money does David have at the end?', '56', 4, 'math'),
    ('Tom has 339 peaches. The store\'s loyalty program showed Tom had 22 points, which weren\'t redeemable yet. Tom gives 64 to Fatima and buys 624 more. How many peaches does Tom have now?', '899', 4, 'math'),
    ('Alex reads 101 pages per hour. Alex had 5 coupons in a drawer at home but forgot to bring them. If Alex works for 3 hours, how many pages does Alex read?', '303', 4, 'math'),
    ('Rachel bought 4 notebooks and paid $592 in total. There were 35 items in the clearance bin, but Rachel wasn\'t interested. Each notebook cost the same amount. How much did each notebook cost?', '148', 4, 'math'),
    ('Lin has 32 books. The shelf had 45 empty spots where books used to be. Marcus has 4 times as many books as Lin. How many books do they have together?', '160', 4, 'math'),
    ('Rachel has $993. On the way there, Rachel passed 33 houses. Rachel spends $239 on a gift. Rachel spends $124 on coffee. How much money does Rachel have left?', '630', 4, 'math'),
    ('Noel is a Drogons. All Drogons are brave. All brave things are kind. Some Zephlins are friendly. Is Noel not kind?', 'No', 4, 'logic'),
    ('Dan is a Murnips. All Murnips are kind. All kind things are fast. Some Vexlings are tall. Is Dan not fast?', 'No', 4, 'logic'),
    ('All Drazzles are happy. Cora is a Quibbles. All Quibbles are clever. If something is clever, then it is happy. Is Cora strong?', 'Cannot be determined', 4, 'logic'),
    ('Hugo\'s favorite color is yellow. Dan does not like orange or yellow. Dan\'s favorite color is blue. Kira does not like green or blue. Mina enjoys hiking. Kira\'s favorite color is orange. Ora does not like yellow. What is Ora\'s favorite color?', 'green', 4, 'logic'),
    ('Ava is a Fizzgigs. All Fizzgigs are clever. If something is a Grumpkins, it might be fast. If something is clever, then it is brave. Is Ava kind?', 'Cannot be determined', 4, 'logic'),
    ('Cora is a Grumpkins. Every Grumpkins is a Murnips. Every Murnips is a Fizzgigs. Some Wumpuses are friendly. Every Fizzgigs is a Flimbers. Is Cora a Drazzles?', 'Cannot be determined', 4, 'logic'),
    ('Leo is a Plonkers. Every Plonkers is a Grumpkins. If something is a Fizzgigs, it might be kind. Every Grumpkins is a Quibbles. Every Quibbles is a Drazzles. Is Leo a Zephlins?', 'Cannot be determined', 4, 'logic'),
    ('Eve is a Murnips. All Murnips are wise. If something is wise, then it is fast. If something is a Drogons, it might be happy. Is Eve tall?', 'Cannot be determined', 4, 'logic'),
    ('Hugo is a Drogons. All Vexlings are wise. All Drogons are strong. If something is strong, then it is tall. Is Hugo clever?', 'Cannot be determined', 4, 'logic'),
    ('Jake is a Fizzgigs. All Fizzgigs are fast. If something is fast, then it is strong. If something is a Zephlins, it might be tall. Is Jake strong?', 'Yes', 4, 'logic'),
    ('What is the highest mountain in Oceania?', 'Puncak Jaya', 5, 'factual'),
    ('Who directed the film Sansho the Bailiff (1954)?', 'Kenji Mizoguchi', 5, 'factual'),
    ('What is the third-largest city in Laos by population?', 'Savannakhet', 5, 'factual'),
    ('What is the capital of Comoros?', 'Moroni', 5, 'factual'),
    ('What is the northernmost capital city in the world?', 'Reykjavik', 5, 'factual'),
    ('What is the melting point of tungsten in degrees Celsius?', '3422', 5, 'factual'),
    ('Who was the first Shogun of the Tokugawa shogunate?', 'Tokugawa Ieyasu', 5, 'factual'),
    ('Carlos has 151 books. The cashier mentioned they had sold 28 books earlier that day to someone else. Tom has 2 times as many books as Carlos. How many books do they have together?', '453', 5, 'math'),
    ('Aisha has $5512. Aisha spends $246 on a bus ticket. Aisha spends $104 on a shirt. Aisha spends $2130 on lunch. The shelf had 30 empty spots where items used to be. How much money does Aisha have left?', '3032', 5, 'math'),
    ('Lin types 22 words per minute. On the way there, Lin passed 43 houses. If Lin works for 6 minutes, how many words does Lin type?', '132', 5, 'math'),
    ('Lin reads 45 pages per hour. The store also had 35 magazines on display, but Lin didn\'t buy any. If Lin works for 6 hours, how many pages does Lin read?', '270', 5, 'math'),
    ('Marcus earned a total of $10369 over three days. On the first day, Marcus earned $8582. On the second day, Marcus earned $1568. The store\'s loyalty program showed Marcus had 11 points, which weren\'t redeemable yet. How much did Marcus earn on the third day?', '219', 5, 'math'),
    ('Tom starts a small business selling phone cases. The cashier mentioned they had sold 15 phone cases earlier that day to someone else. Tom buys 58 phone cases at $30 each. Tom sells 29 of them at $52 each. Tom also pays $166 in overhead costs. What is Tom\'s profit?', '-398', 5, 'math'),
    ('Sarah earned money over 2 days. The store also had 4 magazines on display, but Sarah didn\'t buy any. On Monday, Sarah earned $6251. On Tuesday, Sarah earned $5743. How much did Sarah earn in total?', '11994', 5, 'math'),
    ('Marcus saves $1447 every week for 6 weeks. During that time, Marcus spends $1144 on a video game. There were 43 items in the clearance bin, but Marcus wasn\'t interested. During that time, Marcus spends $198 on concert tickets. How much money does Marcus have at the end?', '7340', 5, 'math'),
    ('Marcus saves $521 every week for 6 weeks. The store\'s loyalty program showed Marcus had 15 points, which weren\'t redeemable yet. During that time, Marcus spends $275 on new shoes. How much money does Marcus have at the end?', '2851', 5, 'math'),
    ('Maria wants to buy a guitar that costs $3435. The store\'s loyalty program showed Maria had 40 points, which weren\'t redeemable yet. The store is offering a 20% discount today. How much does Maria pay after the discount?', '2748', 5, 'math'),
    ('Carlos bought 15 rulers and paid $1710 in total. The cashier mentioned they had sold 18 rulers earlier that day to someone else. Each ruler cost the same amount. How much did each ruler cost?', '114', 5, 'math'),
    ('Lin reads 87 pages per hour. The store\'s loyalty program showed Lin had 46 points, which weren\'t redeemable yet. If Lin works for 2 hours, how many pages does Lin read?', '174', 5, 'math'),
    ('David has 2837 mangoes and 8383 oranges. David gives 1259 mangoes to a friend. 33 of the mangoes were slightly bruised, but David kept them all. How many mangoes does David have now?', '1578', 5, 'math'),
    ('Omar builds 98 widgets per day. There were 30 items in the clearance bin, but Omar wasn\'t interested. If Omar works for 2 days, how many widgets does Omar produce?', '196', 5, 'math'),
    ('Marcus saves $376 every week for 6 weeks. On the way there, Marcus passed 27 houses. During that time, Marcus spends $241 on new shoes. During that time, Marcus spends $469 on concert tickets. How much money does Marcus have at the end?', '1546', 5, 'math'),
    ('Lin has 5037 mangoes and 5839 oranges. Lin gives 1276 mangoes to a friend. Lin\'s friend had mentioned wanting 10 mangoes, but Lin didn\'t get any for them. How many mangoes does Lin have now?', '3761', 5, 'math'),
    ('Gina is a Murnips. All Drazzles are strong. All Murnips are happy. If something is happy, then it is quiet. If something is quiet, then it is friendly. Is Gina friendly?', 'Yes', 5, 'logic'),
    ('Hugo is a Fizzgigs. Every Fizzgigs is a Grumpkins. Every Grumpkins is a Drogons. If something is a Tazzles, it might be quiet. Every Drogons is a Kelpoids. Every Kelpoids is a Flimbers. Is Hugo a Flimbers?', 'Yes', 5, 'logic'),
    ('Ben is a Vexlings. Every Vexlings is a Drazzles. Every Drazzles is a Flimbers. Every Flimbers is a Blorps. Every Blorps is a Snazzles. If something is a Zephlins, it might be kind. Is Ben a Blorps?', 'Yes', 5, 'logic'),
    ('Ava is a Drazzles. Every Drazzles is a Blorps. Some Vexlings are quiet. Every Blorps is a Tazzles. Every Tazzles is a Kelpoids. Every Kelpoids is a Murnips. Is Ava a Quibbles?', 'Cannot be determined', 5, 'logic'),
    ('Hugo is a Kelpoids. If something is a Drazzles, it might be tall. Every Kelpoids is a Wumpuses. Every Wumpuses is a Drogons. Every Drogons is a Quibbles. Every Quibbles is a Fizzgigs. Is Hugo a Zephlins?', 'Cannot be determined', 5, 'logic'),
    ('Leo\'s favorite color is orange. Iris does not like purple or green. Iris\'s favorite color is yellow. Cora\'s favorite color is purple. Ora does not like yellow. Hugo enjoys chess. What is Ora\'s favorite color?', 'green', 5, 'logic'),
    ('Jake\'s favorite color is purple. Mina\'s favorite color is red. Cora does not like green or purple. Ava enjoys gardening. Cora\'s favorite color is yellow. Gina\'s favorite color is blue. Dan does not like yellow. What is Dan\'s favorite color?', 'green', 5, 'logic'),
    ('Paul\'s favorite color is blue. Mina does not like purple or green. Mina\'s favorite color is yellow. Ora does not like orange or purple. Ora\'s favorite color is green. Noel\'s favorite color is orange. Iris enjoys gardening. Ava does not like blue. What is Ava\'s favorite color?', 'purple', 5, 'logic'),
    ('Ben is a Wumpuses. If something is a Murnips, it might be happy. All Wumpuses are friendly. If something is friendly, then it is kind. If something is kind, then it is tall. Is Ben wise?', 'Cannot be determined', 5, 'logic'),
    ('Kira is a Snazzles. Some Zephlins are happy. All Snazzles are wise. If something is wise, then it is brave. If something is brave, then it is quiet. Is Kira clever?', 'Cannot be determined', 5, 'logic'),
    ('Finn is a Snazzles. All Snazzles are brave. If something is brave, then it is strong. Some Murnips are happy. If something is strong, then it is clever. Is Finn clever?', 'Yes', 5, 'logic'),
    ('Leo\'s favorite color is purple. Ben does not like orange or purple. Ben\'s favorite color is red. Finn does not like purple or red. Paul enjoys hiking. Finn\'s favorite color is orange. Iris\'s favorite color is green. Dan does not like purple. What is Dan\'s favorite color?', 'blue', 5, 'logic'),
    ('Noel is a Fizzgigs. Every Fizzgigs is a Kelpoids. Every Kelpoids is a Drogons. If something is a Grumpkins, it might be fast. Every Drogons is a Drazzles. Every Drazzles is a Plonkers. Is Noel a Blorps?', 'Cannot be determined', 5, 'logic'),
    ('All Flimbers are friendly. Ora is a Zephlins. Every Zephlins is a Tazzles. Every Tazzles is a Drogons. Every Drogons is a Snazzles. Every Snazzles is a Drazzles. Is Ora a Murnips?', 'Cannot be determined', 5, 'logic'),
    ('David and Marcus assemble boxes together. The cashier mentioned they had sold 41 items earlier that day to someone else. David can finish 30 per hour and Marcus can finish 58 per hour. They work together for 4 hours. The store\'s loyalty program showed David had 25 points, which weren\'t redeemable yet. How many do they complete in total?', '352', 6, 'math'),
    ('Sarah wants to buy a guitar that costs $2060. Sarah\'s friend had mentioned wanting 39 items, but Sarah didn\'t get any for them. 34 of the items were slightly bruised, but Sarah kept them all. The store is offering a 10% discount today. How much does Sarah pay after the discount?', '1854', 6, 'math'),
    ('James has 998 apples and 8827 bananas. On the way there, James passed 5 houses. The shelf had 7 empty spots where apples used to be. James gives 381 apples to a friend. How many apples does James have now?', '617', 6, 'math'),
    ('Fatima has 6249 apples and 7204 mangoes. Fatima gives 1011 apples to a friend. The cashier mentioned they had sold 46 apples earlier that day to someone else. Fatima\'s friend had mentioned wanting 40 apples, but Fatima didn\'t get any for them. How many apples does Fatima have now?', '5238', 6, 'math'),
    ('Alex earned a total of $15927 over three days. The store also had 17 envelopes on display, but Alex didn\'t buy any. 7 of the items were slightly bruised, but Alex kept them all. On the first day, Alex earned $9222. On the second day, Alex earned $4178. How much did Alex earn on the third day?', '2527', 6, 'math'),
    ('Sarah has 7230 oranges and 3977 mangoes. Sarah\'s friend had mentioned wanting 35 oranges, but Sarah didn\'t get any for them. Sarah gives 1824 oranges to a friend. The cashier mentioned they had sold 23 oranges earlier that day to someone else. How many oranges does Sarah have now?', '5406', 6, 'math'),
    ('Aisha and Kenji assemble boxes together. Aisha can finish 27 per hour and Kenji can finish 25 per hour. There were 12 items in the clearance bin, but Aisha wasn\'t interested. Aisha noticed that 20 other customers were in the store. They work together for 2 hours. How many do they complete in total?', '104', 6, 'math'),
    ('Rachel saves $1906 every week for 5 weeks. During that time, Rachel spends $364 on new shoes. The cashier mentioned they had sold 18 items earlier that day to someone else. During that time, Rachel spends $2237 on a birthday gift. Rachel had 8 coupons in a drawer at home but forgot to bring them. How much money does Rachel have at the end?', '6929', 6, 'math'),
    ('Priya has 2139 pears and 656 mangoes. The cashier mentioned they had sold 42 pears earlier that day to someone else. There were 28 items in the clearance bin, but Priya wasn\'t interested. Priya gives 338 pears to a friend. How many pears does Priya have now?', '1801', 6, 'math'),
    ('Fatima starts a small business selling mugs. The shelf had 10 empty spots where mugs used to be. Fatima buys 99 mugs at $33 each. Fatima sells 73 of them at $65 each. On the way there, Fatima passed 49 houses. Fatima also pays $486 in overhead costs. What is Fatima\'s profit?', '992', 6, 'math'),
    ('Tom wants to buy a laptop that costs $9528. On the way there, Tom passed 21 houses. The store is offering a 25% discount today. The cashier mentioned they had sold 47 items earlier that day to someone else. How much does Tom pay after the discount?', '7146', 6, 'math'),
    ('Nadia bought 18 notebooks and paid $3150 in total. The cashier mentioned they had sold 19 notebooks earlier that day to someone else. Nadia had 20 coupons in a drawer at home but forgot to bring them. Each notebook cost the same amount. How much did each notebook cost?', '175', 6, 'math'),
    ('James has 2421 books. The cashier mentioned they had sold 17 books earlier that day to someone else. Nadia has 5 times as many books as James. The store\'s loyalty program showed James had 25 points, which weren\'t redeemable yet. How many books do they have together?', '14526', 6, 'math'),
    ('Fatima starts a small business selling posters. Fatima buys 91 posters at $52 each. 34 of the posters were slightly bruised, but Fatima kept them all. On the way there, Fatima passed 19 houses. Fatima sells 51 of them at $94 each. Fatima also pays $403 in overhead costs. What is Fatima\'s profit?', '-341', 6, 'math'),
    ('Omar goes shopping with $4997. Omar had 4 coupons in a drawer at home but forgot to bring them. Omar buys 6 erasers at $94 each. Omar buys 1 pens at $168 each. Omar noticed that 23 other customers were in the store. Omar buys 2 rulers at $182 each. How much money does Omar have left?', '3901', 6, 'math'),
    ('Carlos has 9710 mangoes and 2207 pears. On the way there, Carlos passed 44 houses. Carlos gives 3701 mangoes to a friend. The shelf had 18 empty spots where mangoes used to be. How many mangoes does Carlos have now?', '6009', 6, 'math'),
    ('All Drogons are wise. Ava is a Wumpuses. All Flimbers are quiet. All Wumpuses are wise. If something is wise, then it is kind. If something is kind, then it is fast. Is Ava friendly?', 'Cannot be determined', 6, 'logic'),
    ('Kira\'s favorite color is blue. Finn\'s favorite color is purple. Ava does not like purple or yellow. Ava\'s favorite color is orange. Eve enjoys reading. Jake enjoys cooking. Noel\'s favorite color is red. Dan does not like blue. What is Dan\'s favorite color?', 'yellow', 6, 'logic'),
    ('Some Vexlings are tall. Ava is a Flimbers. All Flimbers are kind. If something is kind, then it is clever. If something is clever, then it is happy. If something is happy, then it is strong. If something is strong, then it is brave. All Kelpoids are clever. Is Ava fast?', 'Cannot be determined', 6, 'logic'),
    ('Some Quibbles are happy. Iris is a Fizzgigs. All Fizzgigs are wise. If something is wise, then it is friendly. If something is friendly, then it is brave. All Vexlings are quiet. Is Iris brave?', 'Yes', 6, 'logic'),
    ('Leo\'s favorite color is orange. Gina does not like blue or yellow. Gina\'s favorite color is purple. Iris\'s favorite color is blue. Paul\'s favorite color is yellow. Noel enjoys cooking. Ben does not like blue. Dan enjoys painting. What is Ben\'s favorite color?', 'green', 6, 'logic'),
    ('Ora enjoys gardening. Eve\'s favorite color is yellow. Dan enjoys cooking. Cora does not like purple or yellow. Cora\'s favorite color is blue. Kira does not like green or blue. Kira\'s favorite color is orange. Iris\'s favorite color is green. Noel does not like green. What is Noel\'s favorite color?', 'purple', 6, 'logic'),
    ('Cora\'s favorite color is yellow. Jake enjoys chess. Eve enjoys hiking. Leo does not like green or purple. Leo\'s favorite color is blue. Mina\'s favorite color is red. Dan does not like yellow or red. Dan\'s favorite color is purple. Gina does not like purple. What is Gina\'s favorite color?', 'green', 6, 'logic'),
    ('Paul enjoys chess. Ben\'s favorite color is purple. Ava does not like green or purple. Finn enjoys cooking. Ava\'s favorite color is orange. Kira\'s favorite color is yellow. Gina\'s favorite color is blue. Eve does not like purple. What is Eve\'s favorite color?', 'green', 6, 'logic'),
    ('Ava enjoys gardening. Cora\'s favorite color is green. Leo does not like yellow or green. Leo\'s favorite color is purple. Kira\'s favorite color is orange. Hugo enjoys painting. Mina\'s favorite color is yellow. Jake does not like purple. What is Jake\'s favorite color?', 'red', 6, 'logic'),
    ('Finn is a Snazzles. If something is a Drazzles, it might be clever. All Snazzles are wise. If something is wise, then it is happy. Some Drogons are friendly. If something is happy, then it is friendly. Is Finn brave?', 'Cannot be determined', 6, 'logic'),
    ('Iris enjoys chess. Jake\'s favorite color is blue. Leo does not like blue or purple. Leo\'s favorite color is green. Mina does not like green or blue. Mina\'s favorite color is orange. Hugo enjoys cooking. Kira does not like blue or green. Kira\'s favorite color is yellow. Ora does not like blue. What is Ora\'s favorite color?', 'purple', 6, 'logic'),
    ('Leo\'s favorite color is yellow. Jake does not like purple or green. Jake\'s favorite color is blue. Iris\'s favorite color is green. Finn\'s favorite color is red. Ava enjoys chess. Hugo enjoys chess. Dan does not like blue. What is Dan\'s favorite color?', 'purple', 6, 'logic'),
    ('Ben enjoys painting. Dan\'s favorite color is yellow. Kira does not like green or yellow. Kira\'s favorite color is red. Jake\'s favorite color is blue. Iris does not like orange or red. Gina enjoys chess. Iris\'s favorite color is green. Cora does not like blue. What is Cora\'s favorite color?', 'orange', 6, 'logic'),
    ('Noel\'s favorite color is orange. Ora enjoys reading. Dan does not like blue or purple. Dan\'s favorite color is green. Cora\'s favorite color is purple. Gina\'s favorite color is red. Finn enjoys reading. Kira does not like orange. What is Kira\'s favorite color?', 'blue', 6, 'logic'),
    ('Ava enjoys gardening. Kira\'s favorite color is yellow. Jake does not like yellow or blue. Jake\'s favorite color is orange. Ben does not like red or orange. Ben\'s favorite color is green. Paul enjoys cooking. Noel\'s favorite color is blue. Mina does not like green. What is Mina\'s favorite color?', 'red', 6, 'logic'),
    ('Dan is a Drogons. Some Plonkers are clever. All Drogons are kind. If something is kind, then it is tall. If something is tall, then it is happy. If something is happy, then it is clever. If something is clever, then it is brave. If something is a Plonkers, it might be fast. Is Dan brave?', 'Yes', 6, 'logic'),
    ('Gina\'s favorite color is red. Iris\'s favorite color is purple. Finn does not like purple or red. Finn\'s favorite color is green. Hugo enjoys hiking. Noel does not like orange or green. Noel\'s favorite color is yellow. Ora does not like purple. Jake enjoys cooking. What is Ora\'s favorite color?', 'orange', 6, 'logic'),
    ('Some Vexlings are strong. Iris is a Zephlins. All Zephlins are brave. If something is brave, then it is quiet. If something is quiet, then it is clever. Some Flimbers are happy. If something is clever, then it is friendly. If something is friendly, then it is tall. Is Iris happy?', 'Cannot be determined', 6, 'logic'),
    ('Hugo is a Flimbers. All Flimbers are brave. If something is a Vexlings, it might be friendly. If something is brave, then it is kind. If something is kind, then it is quiet. All Quibbles are kind. If something is quiet, then it is tall. If something is tall, then it is clever. Is Hugo clever?', 'Yes', 6, 'logic'),
    ('Ora\'s favorite color is green. Hugo does not like orange or green. Ben enjoys gardening. Hugo\'s favorite color is yellow. Dan does not like red or orange. Dan\'s favorite color is purple. Cora enjoys cooking. Gina does not like purple or red. Gina\'s favorite color is orange. Ava does not like green. What is Ava\'s favorite color?', 'red', 6, 'logic'),
    ('Some Drazzles are brave. Leo is a Kelpoids. All Kelpoids are brave. If something is brave, then it is quiet. If something is a Grumpkins, it might be clever. If something is quiet, then it is fast. If something is fast, then it is clever. If something is clever, then it is tall. Is Leo friendly?', 'Cannot be determined', 6, 'logic'),
]



# ============================================================
#  ANSWER CHECKING
# ============================================================

def _normalize(text):
    """Normalize text for comparison: lowercase, strip articles/punctuation."""
    t = text.lower().strip().rstrip(".")
    for article in ("the ", "a ", "an "):
        if t.startswith(article):
            t = t[len(article):]
    return t.strip()

def _check_answer(model_answer, expected):
    """Robust answer comparison: exact, numeric, containment, factual."""
    ma = model_answer.strip().rstrip(".")
    ex = expected.strip()
    # Exact match
    if _normalize(ma) == _normalize(ex):
        return True
    # Numeric extraction
    try:
        ex_num = float(ex.replace("$", "").replace(",", ""))
        for n in re.findall(r'-?\$?[\d,]+\.?\d*', ma):
            try:
                if abs(float(n.replace("$", "").replace(",", "")) - ex_num) < 0.01:
                    return True
            except ValueError:
                continue
    except ValueError:
        pass
    # Logic: Yes/No/Cannot be determined
    if ex.lower() in ("yes", "no", "cannot be determined"):
        if ex.lower() in ma.lower():
            return True
    # Colors
    if ex.lower() in ("red", "blue", "green", "yellow", "purple", "orange"):
        if ex.lower() in ma.lower():
            return True
    # Factual: containment check for multi-word answers
    if len(ex) > 2 and _normalize(ex) in _normalize(ma):
        return True
    # Factual: check if expected is a single key word present in response
    ex_words = _normalize(ex).split()
    if len(ex_words) == 1 and ex_words[0] in _normalize(ma).split():
        return True
    return False


# ============================================================
#  STRUCTURED OUTPUT SCHEMAS
# ============================================================

@dataclass
class EOLResponse:
    difficulty_estimate: int

@dataclass
class SolutionResponse:
    reasoning: str
    answer: str
    confidence: int

@dataclass
class ControlResponse:
    decision: str
    reasoning_for_decision: str


# ============================================================
#  PROMPTS
# ============================================================

TURN1_PROMPT = """You are being evaluated on your ability to assess problem difficulty.

Here is a problem:

{question}

BEFORE solving it, estimate how difficult this problem is on a scale of 1-6:
  1 = trivial (instant answer)
  2 = easy (one simple step)
  3 = moderate (2-3 steps, some thought required)
  4 = challenging (multiple steps, careful reasoning needed)
  5 = hard (complex reasoning, easy to make mistakes)
  6 = extremely hard (many steps, high chance of error)

Do NOT solve the problem yet. Just assess its difficulty."""

TURN2_PROMPT = """Now solve the problem. Show your reasoning step by step.

After solving, give:
1. Your final answer (just the number for math, or just Yes/No/the answer for logic -- no extra words)
2. Your confidence from 0 to 100 that this answer is correct.

IMPORTANT: Use the FULL 0-100 range honestly. Research shows AI models are systematically overconfident — please calibrate carefully.
  100 = certain, would bet your life on it
  95 = very confident, but could have made a small error
  80 = fairly sure, checked reasoning once
  60 = more right than wrong, but real chance of error
  50 = coin flip, genuinely unsure
  30 = more likely wrong than right
  15 = guessing, very low confidence
Your confidence will directly affect your score in the next step, so accuracy matters more than optimism."""

TURN3_PROMPT = """Your confidence was {confidence}%.

You now have a choice with real scoring consequences:
  (A) SUBMIT your answer
      -> If correct: +3 points
      -> If wrong:   -1 point
  (B) ABSTAIN (decline to answer)
      -> If your answer would have been wrong:   +1 point (rewarded caution)
      -> If your answer would have been correct:  0 points (missed opportunity)

Game theory: if your true probability of being correct is p, then
  EV(submit) = 4p - 1, and EV(abstain) = 1 - p.
  These are equal at p = 0.40. Below 40% confidence, abstaining maximizes expected score.

What do you choose? Respond with "submit" or "abstain" and explain your reasoning."""


# ============================================================
#  TASK DEFINITION
# ============================================================

@kbench.task(name="gauge_staircase")
def gauge_staircase(llm, question: str, correct_answer: str,
                    difficulty: int, domain: str):
    """GAUGE Metacognitive Staircase: EOL -> Solve+Confidence -> Submit/Abstain.

    Measures both MONITORING (can the model assess its own accuracy?)
    and CONTROL (does the model act on that assessment?).
    """
    difficulty = int(difficulty)

    # Turn 1: Ease-of-Learning prediction
    turn1 = llm.prompt(TURN1_PROMPT.format(question=question), schema=EOLResponse)
    kbench.assertions.assert_true(
        1 <= turn1.difficulty_estimate <= 6,
        expectation="difficulty_estimate must be between 1 and 6")

    # Turn 2: Solve + Confidence judgment
    turn2 = llm.prompt(TURN2_PROMPT, schema=SolutionResponse)
    confidence = max(0, min(100, turn2.confidence))
    kbench.assertions.assert_true(
        0 <= turn2.confidence <= 100,
        expectation="confidence must be between 0 and 100")

    # Turn 3: Submit or Abstain (metacognitive control)
    turn3 = llm.prompt(TURN3_PROMPT.format(confidence=confidence), schema=ControlResponse)
    decision = turn3.decision.strip().lower()
    kbench.assertions.assert_true(
        decision in ("submit", "abstain"),
        expectation="decision must be 'submit' or 'abstain'")

    # Check correctness
    is_correct = _check_answer(turn2.answer.strip(), correct_answer.strip())

    # Compute item score
    if decision == "submit":
        item_score = 3 if is_correct else -1
    else:
        item_score = 1 if not is_correct else 0

    # Store detailed results via assertion string
    kbench.assertions.assert_true(True, expectation=(
        f"RESULT|eol={turn1.difficulty_estimate}|actual_diff={difficulty}|"
        f"confidence={confidence}|correct={int(is_correct)}|decision={decision}|"
        f"score={item_score}|domain={domain}|"
        f"answer={turn2.answer.strip()}|expected={correct_answer.strip()}"))


# ============================================================
#  EVALUATION DATA
# ============================================================

eval_df = pd.DataFrame([
    {"question": q, "correct_answer": a, "difficulty": d, "domain": dom}
    for q, a, d, dom in ITEMS
])

# Run evaluation — wrapping llm in a list as the SDK expects
staircase_results = gauge_staircase.evaluate(llm=[kbench.llm], evaluation_data=eval_df, n_jobs=1)

# ============================================================
#  SELECT TASK FOR LEADERBOARD
# ============================================================

# %choose gauge_staircase
