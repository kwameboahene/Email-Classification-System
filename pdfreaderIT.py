
import fitz
import os
json = []
import tojson

#goes through files in the directory
for filename in os.listdir(os.getcwd()):
    #if the file is a pdf file
    if(filename.endswith('.pdf')):
        #open it
        file = fitz.open(filename)
        p = file.loadPage(0)
        text = p.getText()
        #split files based on new line character
        lines = text.split('\n')

        diction = {"From":"","Subject":"","Message":"","Class":"0"}
        
        Message = ""

        
        for i in range(0,len(lines)):
            # if line is says From:
            if lines[i] == "From:":
                # assign dictionary from to next line
                diction["From"] = lines[i+1]

            # if line is from Subject:
            elif lines[i] == "Subject:":
                # assign dictionary subject to next line
                diction["Subject"] = lines[i+1]
                
                # the remaining lines are the body of the email
                for m in range(i+2, len(lines)):
                    # concatenate everything to message
                    Message = Message + lines[m]

        #check if message is empty assign message to subject  
        if Message == "":
            diction["Message"] = diction["Subject"]

        elif Message == " ":
            diction["Message"] = diction["Subject"]

        elif Message == "  ":
            diction["Message"] = diction["Subject"]

        elif Message == "   ":
            diction["Message"] = diction["Subject"]
            
        #else message is Message                
        else:
            diction["Message"] = Message
                    
                    
        #email to json
        json.append(diction)
        print(diction)

#write json to json       
tojson.writeToJson("IT.json",json)

