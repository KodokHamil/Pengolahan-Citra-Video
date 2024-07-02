import cv2
import numpy as np
from tensorflow import keras

def detect_card(frame, draw=True):
    original = frame.copy()
    green_lower = np.array([40, 240, 240])
    green_upper = np.array([80, 255, 255])
    hsv = cv2.cvtColor(original, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, green_lower, green_upper)
    mask = cv2.bitwise_not(mask)
    greenscreen = cv2.bitwise_and(original, original, mask=mask)
    edges = cv2.Canny(greenscreen, 50, 150)
    contours, _ = cv2.findContours(edges.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    card_corners = []

    # Validated area parameters (adjust these as needed)
    x1, y1, x2, y2 = 200, 100, 400, 400

    for contour in contours:
        area = cv2.contourArea(contour)
        if area < 1000:
            continue

        perimeter = cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, 0.025 * perimeter, True)
        if len(approx) == 4:
            # Check if contour is within validated area
            valid_contour = all(x1 <= point[0][0] <= x2 and y1 <= point[0][1] <= y2 for point in approx)
            if not valid_contour:
                continue

            card_corners.append(approx)
            if draw:
                for i in range(len(approx)):
                    start_point = tuple(approx[i][0])
                    end_point = tuple(approx[(i + 1) % len(approx)][0])
                    cv2.line(original, start_point, end_point, (255, 0, 255), 2)
                    cv2.circle(original, (approx[i][0][0], approx[i][0][1]), 3, (255, 0, 0), 3)

    # Draw the validated area rectangle
    cv2.rectangle(original, (x1, y1), (x2, y2), (0, 255, 0), 2)

    # Add text to prompt user to press 'e' for scanning
    cv2.putText(original, "Tekan 'e' untuk scan kartu", (150, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 1)
    cv2.putText(original, "Tekan 's' untuk ambil dari stack", (120, 450), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 1)
    cv2.imshow('Kartu Terdeteksi', original)
    cv2.waitKey(1)

    return card_corners

def warp(frame, corners):
    corners = sorted(corners, key=lambda x: x[0][0] + x[0][1])
    pts1 = np.float32([corners[0][0], corners[1][0], corners[2][0], corners[3][0]])
    width = max(abs(corners[0][0][0] - corners[1][0][0]), abs(corners[2][0][0] - corners[3][0][0]))
    height = max(abs(corners[0][0][1] - corners[2][0][1]), abs(corners[1][0][1] - corners[3][0][1]))
    pts2 = np.float32([[0, 0], [width, 0], [0, height], [width, height]])
    matrix = cv2.getPerspectiveTransform(pts1, pts2)
    imgOutput = cv2.warpPerspective(frame, matrix, (width, height))
    cv2.imshow('Kartu Wrapped', imgOutput)
    cv2.waitKey(1)
    return imgOutput

def preprocess_image_for_model(image):
    image = cv2.resize(image, (128, 128))
    image = np.asarray(image) / 255.0
    image = image.astype('float32')
    image = np.expand_dims(image, axis=0)
    return image

def recognize_card(image, model, label_classes):
    processed_image = preprocess_image_for_model(image)
    prediction = model.predict(processed_image)
    class_idx = np.argmax(prediction)
    card = label_classes[class_idx]
    return card

def get_card_value(card):
    if card[1:].isdigit():
        value = int(card[1:])
        if value > 10:  # J, Q, K cases
            return 10
        if value == 1:  # untuk Ace
            return 11
        return value
    else:
        return 10  # Default case for safety

def calculate_total(player_cards):
    total = 0
    num_aces = 0
    
    for card in player_cards:
        value = get_card_value(card)
        if value == 11:
            num_aces += 1
        total += value
    
    while total > 41 and num_aces:
        total -= 10
        num_aces -= 1
    
    return total

def calculate_total_by_suit(player_cards):
    suit_totals = {'D': 0, 'C': 0, 'H': 0, 'S': 0}
    suit_aces = {'D': 0, 'C': 0, 'H': 0, 'S': 0}

    for card in player_cards:
        suit = card[0]
        value = get_card_value(card)
        if value == 11:
            suit_aces[suit] += 1
        suit_totals[suit] += value
    
    for suit in suit_totals:
        while suit_totals[suit] > 41 and suit_aces[suit]:
            suit_totals[suit] -= 10
            suit_aces[suit] -= 1

    return max(suit_totals.values())

def discard_card(player_cards, new_card, discard_stack):
    suits_count = {'D': 0, 'C': 0, 'H': 0, 'S': 0}
    for card in player_cards:
        suits_count[card[0]] += 1
    suits_count[new_card[0]] += 1

    # Cari kartu yang mau di discard
    min_value = float('inf')
    card_to_discard = None
    for card in player_cards:
        if get_card_value(card) < min_value and card[0] != new_card[0]:
            min_value = get_card_value(card)
            card_to_discard = card

    if card_to_discard is None:
        # jika kartu tidak ada suit yang berbeda, buang yang nilainya terkecil
        min_value = float('inf')
        for card in player_cards:
            if get_card_value(card) < min_value:
                min_value = get_card_value(card)
                card_to_discard = card

    if card_to_discard:
        player_cards.remove(card_to_discard)
        discard_stack.append(card_to_discard)

def print_deck(player_cards, computer_cards):
    print("\nDeck Pemain:")
    for card in player_cards:
        print(card, end=" ")
    print("\nDeck Komputer:")
    for card in computer_cards:
        print(card, end=" ")
    print("\n")

def print_discard_stack(discard_stack):
    print("\nStack Kartu yang Dibuang:")
    for card in discard_stack:
        print(card, end=" ")
    print("\n")

def main():
    cap = cv2.VideoCapture(0)
    model = keras.models.load_model('Cards.h5')
    label_classes = ("D1", "D2", "D3", "D4", "D5", "D6", "D7", "D8", "D9", "D10", "D11", "D12", "D13",
                     "C1", "C2", "C3", "C4", "C5", "C6", "C7", "C8", "C9", "C10", "C11", "C12", "C13",
                     "H1", "H2", "H3", "H4", "H5", "H6", "H7", "H8", "H9", "H10", "H11", "H12", "H13",
                     "S1", "S2", "S3", "S4", "S5", "S6", "S7", "S8", "S9", "S10", "S11", "S12", "S13")

    if not cap.isOpened():
        print("Error: Kamera tidak dapat diakses.")
        exit()

    player_cards = []
    computer_cards = []
    discard_stack = []

    def draw_card(deck):
        if len(deck) < 1:
            return None
        return deck.pop()

    # Initialize deck
    deck = list(label_classes)
    np.random.shuffle(deck)

    # Deal initial cards
    for _ in range(4):
        player_cards.append(draw_card(deck))
        computer_cards.append(draw_card(deck))

    def scan_card(deck, scanned_cards):
        while True:
            ret, frame = cap.read()
            if not ret:
                print("Error: Frame tidak dapat diambil dari kamera.")
                break

            card_corners = detect_card(frame, draw=True)
            if card_corners and cv2.waitKey(1) & 0xFF == ord('e'):
                for corners in card_corners:
                    warped_card = warp(frame, corners)
                    card = recognize_card(warped_card, model, label_classes)
                    if card not in scanned_cards and card in deck:
                        deck.remove(card)
                        return card

    print_deck(player_cards, computer_cards)
    
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Frame tidak dapat diambil dari kamera.")
            break

        card_corners = detect_card(frame, draw=True)

        if cv2.waitKey(1) & 0xFF == ord('s'):
            if discard_stack:
                card = discard_stack.pop()
                if len(player_cards) >= 4:
                    discard_card(player_cards, card, discard_stack)
                player_cards.append(card)
                print_deck(player_cards, computer_cards)
                print_discard_stack(discard_stack)

                player_total = calculate_total_by_suit(player_cards)
                print(f'Kartu Terdeteksi: {card} (Nilai: {get_card_value(card)})')
                print(f'Total Poin Player: {player_total}')

                if player_total == 41:
                    print("Pemain menang! Anda mencapai 41 tepat!")
                    cap.release()
                    cv2.destroyAllWindows()
                    return

                # Computer's turn
                computer_card = discard_stack.pop() if discard_stack else scan_card(deck, player_cards + computer_cards)
                if computer_card:
                    if len(computer_cards) >= 4:
                        discard_card(computer_cards, computer_card, discard_stack)
                    computer_cards.append(computer_card)
                    computer_total = calculate_total_by_suit(computer_cards)
                    print_deck(player_cards, computer_cards)
                    print_discard_stack(discard_stack)
                    print(f'Total Poin Komputer: {computer_total}')

                    if computer_total == 41:
                        print("Komputer menang! Komputer mencapai 41 tepat!")
                        cap.release()
                        cv2.destroyAllWindows()
                        return

        elif card_corners and cv2.waitKey(1) & 0xFF == ord('e'):
            for corners in card_corners:
                warped_card = warp(frame, corners)
                card = recognize_card(warped_card, model, label_classes)

                if card in player_cards or card in computer_cards:
                    print(f'Kartu {card} sudah ada di deck, scan ulang.')
                    continue

                # Remove a card before drawing a new one if the count exceeds
                if len(player_cards) >= 4:
                    discard_card(player_cards, card, discard_stack)

                player_cards.append(card)
                print_deck(player_cards, computer_cards)
                print_discard_stack(discard_stack)

                player_total = calculate_total_by_suit(player_cards)
                print(f'Kartu Terdeteksi: {card} (Nilai: {get_card_value(card)})')
                print(f'Total Poin Player: {player_total}')

                if player_total == 41:
                    print("Pemain menang! Anda mencapai 41 tepat!")
                    cap.release()
                    cv2.destroyAllWindows()
                    return

                # Computer's turn
                computer_card = scan_card(deck, player_cards + computer_cards)
                if computer_card:
                    if len(computer_cards) >= 4:
                        discard_card(computer_cards, computer_card, discard_stack)
                    computer_cards.append(computer_card)
                    computer_total = calculate_total_by_suit(computer_cards)
                    print_deck(player_cards, computer_cards)
                    print_discard_stack(discard_stack)
                    print(f'Total Poin Komputer: {computer_total}')

                    if computer_total == 41:
                        print("Komputer menang! Komputer mencapai 41 tepat!")
                        cap.release()
                        cv2.destroyAllWindows()
                        return

        elif cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
