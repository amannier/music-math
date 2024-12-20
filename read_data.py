import musicpy as mp
import os

def createMbpFile():
    filenames = os.listdir("data/midi")
    pieces = []
    for filename in filenames:
        current_piece = mp.read(f"data/midi/{filename}")
        pieces.append(current_piece)
    mp.write_data(pieces, name="data/pieces.mbp")

def readPiecesAndExtractMelody():
    pieces = mp.load_data("data/pieces.mbp")
    melodies = []
    for piece in pieces:
        a, bpm, strat_time = piece.merge()
        melody = a.split_melody(mode='chord')
        melodies.append(melody)
    return melodies

def readMainTracks():
    chords = mp.load_data("data/chords.mbp")
    for i, chord in enumerate(chords):
        chords[i] = chord
    return chords

def readBars():
    bars = mp.load_data("data/bars.mbp")
    return bars

def melody2Bars(melodyList):
    allBars = []
    for melody in melodyList:
        bars = melody.split_bars()
        for bar in bars:
            allBars.append(bar)
        # end for bars

        '''
        # 提取出不重复的小节
        for i, i_bar in enumerate(bars):
            # 不要太短的小节
            if len(i_bar) < 3:
                continue
            unique = True
            i_bar_names = i_bar.names()
            i_name = ''
            for char in i_bar_names:
                i_name += char
            # end for char i
            for j, j_bar in enumerate(bars):
                if i == j:
                    continue
                # else
                j_bar_names = j_bar.names()
                j_name = ''
                for char in j_bar_names:
                    j_name += char
                # end for char j
                if i_name == j_name:
                    unique == False
                    break
                # end if
            # end for j
            if unique:
                allBars.append(i_bar)
            # end if unique
        # end for i
        '''

    # end for melody
    return set(allBars)
# end def melody2Bars

def writeBars():
    melodies = readPiecesAndExtractMelody()
    # melodies = readMainTracks()
    allBars = melody2Bars(melodies)
    mp.write_data(allBars, name="data/bars.mbp")
    return

def writeFirstTrack(pieces):
    filenames = os.listdir("data/midi")
    for i, piece in enumerate(pieces):
        filename = filenames[i].strip('.mid') + ' track0.mid'
        mp.write(piece(0), name=f"data/track0/{filename}")
    # end for
    return

# if __name__ == '__main__':
#     # writeBars()
#     allBars = readBars()
#     for bar in allBars:
#         print(bar.notes)
