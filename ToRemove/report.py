from fpdf import FPDF
from PIL import Image
from auth import authenticate_face

class PDFReport(FPDF):
    def header(self):
        self.set_font('Helvetica', 'B', 12)
        self.cell(0, 10, 'Face Authentication Report', new_x='LMARGIN', new_y='NEXT', align='C')
        self.ln(10)

    def footer(self):
        self.set_y(-15)
        self.set_font('Helvetica', 'I', 8)
        self.cell(0, 10, f'Page {self.page_no()}', align='C')
        
    def add_image_statistics(self, matches_count, non_matches_count):
        """ Add statistics for each individual image analyzed. """
        self.set_font('Helvetica', 'I', 9)
        total_attempts = matches_count + non_matches_count
        accuracy = (matches_count / total_attempts) * 100 if total_attempts else 0
        self.cell(0, 8, f'Statistics for this image:', new_x='LMARGIN', new_y='NEXT')
        self.cell(0, 8, f'Matches: {matches_count}', new_x='LMARGIN', new_y='NEXT')
        self.cell(0, 8, f'Non-Matches: {non_matches_count}', new_x='LMARGIN', new_y='NEXT')
        self.cell(0, 8, f'Accuracy: {accuracy:.2f}%', new_x='LMARGIN', new_y='NEXT')
        self.ln(5)


    def add_image_to_pdf(self, image_path, w=40):
        """ Helper function to insert an image into the PDF with a specific width. """
        try:
            with Image.open(image_path) as img:
                img_width, img_height = img.size
                aspect_ratio = img_height / img_width
                h = w * aspect_ratio
                self.image(image_path, w=w, h=h)
        except Exception as e:
            print(f"Error inserting image {image_path}: {e}")



    def add_authentication_result(self, image_path, matches, non_matches):
        # Display the main image being analyzed
        self.set_font('Helvetica', 'B', 10)
        self.cell(0, 10, f'Results for Image: {image_path}', new_x='LMARGIN', new_y='NEXT')
        self.add_image_to_pdf(image_path, w=50)  # Show the main image being analyzed
        self.ln(10)  # Add space after the main image

        # Display matches with images
        if matches:
            self.set_font('Helvetica', 'B', 9)
            self.cell(0, 10, 'Matches:', new_x='LMARGIN', new_y='NEXT')
            self.set_font('Helvetica', '', 9)
            for match in matches:
                self._set_status_color(match[3])
                self.cell(0, 8, f'Matched with {match[0]} (Image: {match[1]}, Distance: {match[2]:.4f}, Status: {self._status_text(match[3])})', new_x='LMARGIN', new_y='NEXT')
                self.add_image_to_pdf(match[1], w=40)  # Show the matched image
                self.set_text_color(0, 0, 0)  # Reset text color to black
                self.ln(5)  # Add space after each match

        # Display non-matches without images
        if non_matches:
            self.set_font('Helvetica', 'B', 9)
            self.cell(0, 10, 'Non-matches:', new_x='LMARGIN', new_y='NEXT')
            self.set_font('Helvetica', '', 9)
            for non_match in non_matches:
                self._set_status_color(non_match[3])
                self.cell(0, 8, f'Not matched with {non_match[0]} (Image: {non_match[1]}, Distance: {non_match[2]:.4f}, Status: {self._status_text(non_match[3])})', new_x='LMARGIN', new_y='NEXT')
                self.add_image_to_pdf(non_match[1], w=40)  # Show the matched image
                self.set_text_color(0, 0, 0)  # Reset text color to black
                self.ln(10)  # Add space after non-matches section
        # Add per-image statistics
        self.add_image_statistics(len(matches), len(non_matches))
        self.ln(10)

    def _set_status_color(self, success):
        if success:
            self.set_text_color(0, 128, 0)  # Green for success
        else:
            self.set_text_color(255, 0, 0)  # Red for failure

    def _status_text(self, success):
        return "Success" if success else "Failed"

    def add_statistics(self, total_images, total_matches, total_non_matches):
        self.set_font('Helvetica', 'B', 10)
        self.cell(0, 10, '--- Statistics ---', new_x='LMARGIN', new_y='NEXT')
        self.set_font('Helvetica', '', 9)
        
        total_attempts = total_matches + total_non_matches
        accuracy = (total_matches / total_attempts) * 100 if total_attempts else 0
        
        self.cell(0, 8, f'Total Images Processed: {total_images}', new_x='LMARGIN', new_y='NEXT')
        self.cell(0, 8, f'Total Matches: {total_matches}', new_x='LMARGIN', new_y='NEXT')
        self.cell(0, 8, f'Total Non-Matches: {total_non_matches}', new_x='LMARGIN', new_y='NEXT')
        self.cell(0, 8, f'Accuracy: {accuracy:.2f}%', new_x='LMARGIN', new_y='NEXT')
        self.ln(10)

# Generate PDF report for a list of input images
def generate_pdf_report(input_images):
    pdf = PDFReport()
    pdf.add_page()

    total_matches = 0
    total_non_matches = 0
    total_images = len(input_images)
    for image_path in input_images:
        matches, non_matches = authenticate_face(image_path)
        print(matches)
        print(non_matches)
    
        # Only add results if there are matches or non-matches
        if matches or non_matches:
            pdf.add_authentication_result(image_path, matches, non_matches)
        else:
            print(f"Skipping {image_path} due to no valid face encodings.")
    
        total_matches += len(matches)
        total_non_matches += len(non_matches)

    # Add statistics at the end
    pdf.add_statistics(total_images, total_matches, total_non_matches)

    pdf_file = input("Insert the name of the report: ")
    pdf.output(pdf_file)
    print(f"PDF Report generated as {pdf_file}")
