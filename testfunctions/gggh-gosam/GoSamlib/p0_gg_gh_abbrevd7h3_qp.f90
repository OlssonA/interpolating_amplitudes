module     p0_gg_gh_abbrevd7h3_qp
   use p0_gg_gh_config, only: ki => ki_qp
   use p0_gg_gh_kinematics_qp, only: epstensor
   use p0_gg_gh_globalsh3_qp
   implicit none
   private
   complex(ki), dimension(19), public :: abb7
   complex(ki), public :: R2d7
   public :: init_abbrev
   complex(ki), parameter :: i_ = (0.0_ki, 1.0_ki)
contains
   subroutine     init_abbrev()
      use p0_gg_gh_config, only: deltaOS, &
     &    logfile, debug_nlo_diagrams
      use p0_gg_gh_kinematics_qp
      use p0_gg_gh_model_qp
      use p0_gg_gh_color_qp, only: TR
      use p0_gg_gh_globalsl1_qp, only: epspow
      implicit none
      abb7(1)=sqrt(mT**2)
      abb7(2)=sqrt2**(-1)
      abb7(3)=spak1k2**(-1)
      abb7(4)=spak2k3**(-1)
      abb7(5)=c1-c2
      abb7(6)=abb7(5)*abb7(3)*abb7(4)*abb7(1)**3
      abb7(7)=i_*e*gHT*abb7(2)
      abb7(8)=abb7(7)*spbk3k1
      abb7(9)=abb7(8)*abb7(6)
      abb7(10)=-abb7(3)*abb7(1)*abb7(5)*abb7(7)
      abb7(11)=abb7(10)*abb7(4)
      abb7(12)=-spbk3k1*abb7(11)
      abb7(13)=es12+es23
      abb7(14)=abb7(12)*abb7(13)
      abb7(14)=-2.0_ki*abb7(9)+abb7(14)
      abb7(14)=es12*abb7(14)
      abb7(9)=-es23*abb7(9)
      abb7(9)=abb7(9)+abb7(14)
      abb7(9)=2.0_ki*abb7(9)
      abb7(14)=abb7(12)*es12
      abb7(15)=-4.0_ki*abb7(14)
      abb7(16)=4.0_ki*abb7(12)
      abb7(16)=es23*abb7(16)
      abb7(17)=8.0_ki*abb7(12)
      abb7(14)=-2.0_ki*abb7(14)
      abb7(18)=abb7(4)**2
      abb7(5)=abb7(1)*abb7(18)*abb7(5)*abb7(8)
      abb7(8)=2.0_ki*es23
      abb7(19)=abb7(5)*abb7(8)
      abb7(5)=-12.0_ki*abb7(5)
      abb7(13)=abb7(11)*abb7(13)
      abb7(6)=abb7(7)*abb7(6)
      abb7(6)=2.0_ki*abb7(6)+abb7(13)
      abb7(6)=2.0_ki*spbk2k1*abb7(6)
      abb7(7)=4.0_ki*spbk2k1
      abb7(7)=-abb7(7)*abb7(11)
      abb7(10)=-abb7(18)*abb7(10)
      abb7(11)=-abb7(10)*abb7(8)
      abb7(13)=mH**2
      abb7(8)=abb7(13)-abb7(8)-es12
      abb7(8)=4.0_ki*abb7(10)*abb7(8)
      abb7(10)=8.0_ki*abb7(10)
      abb7(12)=2.0_ki*abb7(12)
      R2d7=0.0_ki
      rat2 = rat2 + R2d7
      if (debug_nlo_diagrams) then
          write (logfile,*) "<result name='r2' index='7' value='", &
          & R2d7, "'/>"
      end if
   end subroutine
end module p0_gg_gh_abbrevd7h3_qp
