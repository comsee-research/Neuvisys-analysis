import rosbag
from bitarray import bitarray

def convert_ros_to_aedat(bag_file, aedat_file, x_size, y_size):
    print("\nFormatting: .rosbag -> .aedat\n")
    
    # open the file and write the headers
    with open(aedat_file, "wb") as file:
        file.write(b'#!AER-DAT2.0\r\n')
        file.write(b'# This is a raw AE data file created by saveaerdat.m\r\n')
        file.write(b'# Data format is int32 address, int32 timestamp (8 bytes total), repeated for each event\r\n')
        file.write(b'# Timestamps tick is 1 us\r\n')
        file.write(b'# End of ASCII Header\r\n')
        
        bag = rosbag.Bag(bag_file)
        
        # setup the camera width and height by adding one event
        for topic, msg, t in bag.read_messages(topics=['/cam0/events']):
            for e in msg.events:
                y = format(y_size-1, "09b")
                x = format(x_size-1, "010b")
                p = "10"
                address = bitarray("0" + y + x + p + "0000000000")
                ts = int(e.ts.to_nsec() / 1000.0)
                timestamp = bitarray(format(ts, "032b"))
                
                file.write(address.tobytes())
                file.write(timestamp.tobytes())
                break
            break

        # format and write the bag content to the aedat file
        for topic, msg, t in bag.read_messages(topics=['/cam0/events']):
            for e in msg.events:
                y = format(y_size-1-e.y, "09b")
                x = format(x_size-1-e.x, "010b")
                p = "10" if e.polarity else "00"
                address = bitarray("0" + y + x + p + "0000000000")
                ts = int(e.ts.to_nsec() / 1000.0)
                timestamp = bitarray(format(ts, "032b"))
                
                file.write(address.tobytes())
                file.write(timestamp.tobytes())
        bag.close()
        
bag_file = "/home/thomas/Desktop/Event/files/out.bag"
aedat_file = "/home/thomas/Desktop/Event/files/out.aedat"
convert_ros_to_aedat(bag_file, aedat_file, 240, 180)