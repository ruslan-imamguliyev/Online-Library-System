package com.project.e_library.entity;

import jakarta.persistence.*;
import lombok.AccessLevel;
import lombok.Getter;
import lombok.Setter;

import java.util.List;

@Getter
@Setter(AccessLevel.NONE)
@Entity
@Table(name = "library")
public class Book {

    @Id
    @GeneratedValue(strategy = GenerationType.IDENTITY)
    @Column(name = "id")
    private int id;

    @Column(name = "title")
    private String title;

    @Column(name = "author")
    private String author;

    @ElementCollection
    @CollectionTable(name = "book_genres", joinColumns = @JoinColumn(name = "book_id"))
    @Column(name = "genre")
    private List<String> genres;

    @Column(name = "description")
    private String description;

    @Column(name = "coverImg")
    private String imgUrl;

    @Column(name = "likedPercent")
    private double ratingPercent;

    private double rating;

    @Column(name = "numRatings")
    private long ratingNumber;

}
