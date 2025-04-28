module     p0_gg_gh_abbrevd1h0
   use p0_gg_gh_config, only: ki
   use p0_gg_gh_kinematics, only: epstensor
   use p0_gg_gh_globalsh0
   implicit none
   private
   complex(ki), dimension(24), public :: abb1
   complex(ki), public :: R2d1
   public :: init_abbrev
   complex(ki), parameter :: i_ = (0.0_ki, 1.0_ki)
contains
   subroutine     init_abbrev()
      use p0_gg_gh_config, only: deltaOS, &
     &    logfile, debug_nlo_diagrams
      use p0_gg_gh_kinematics
      use p0_gg_gh_model
      use p0_gg_gh_color, only: TR
      use p0_gg_gh_globalsl1, only: epspow
      implicit none
      abb1(1)=sqrt(mT**2)
      abb1(2)=sqrt2**(-1)
      abb1(3)=es12**(-1)
      abb1(4)=spbk2k1**(-1)
      abb1(5)=spak2k3**(-1)
      abb1(6)=spbk3k2**(-1)
      abb1(7)=spak2l4**(-1)
      abb1(8)=spbl4k2**(-1)
      abb1(9)=c1-c2
      abb1(10)=i_*e*gHT*abb1(5)*abb1(4)*abb1(2)
      abb1(11)=abb1(10)*abb1(3)*abb1(1)
      abb1(12)=-abb1(11)*abb1(9)
      abb1(13)=spak1k2**2
      abb1(14)=abb1(13)*spbk3k1
      abb1(15)=-abb1(14)*abb1(12)
      abb1(16)=2.0_ki*abb1(15)
      abb1(11)=abb1(9)*abb1(11)*spak1k2
      abb1(17)=abb1(11)*spak2l4
      abb1(18)=abb1(17)*spbl4k3
      abb1(19)=mH**2*abb1(8)*abb1(7)
      abb1(20)=abb1(19)*abb1(15)
      abb1(18)=abb1(18)-abb1(20)
      abb1(20)=-spbk3k1*abb1(11)
      abb1(21)=abb1(20)*abb1(6)
      abb1(22)=spak1l4*spbl4k3
      abb1(23)=abb1(22)*abb1(21)
      abb1(24)=-abb1(23)+abb1(18)
      abb1(24)=es23*abb1(24)
      abb1(9)=-abb1(10)*abb1(9)
      abb1(10)=-abb1(9)*abb1(1)**3*abb1(14)*abb1(3)
      abb1(10)=2.0_ki*abb1(10)+abb1(24)
      abb1(10)=2.0_ki*abb1(10)
      abb1(14)=2.0_ki*abb1(23)-abb1(18)
      abb1(14)=4.0_ki*abb1(14)
      abb1(15)=-12.0_ki*abb1(15)
      abb1(18)=-spbl4k2*abb1(6)*abb1(16)
      abb1(12)=-abb1(13)*abb1(12)
      abb1(13)=spbl4k1*abb1(12)
      abb1(13)=abb1(18)+abb1(13)
      abb1(13)=2.0_ki*abb1(13)
      abb1(18)=spak1l4*abb1(20)
      abb1(20)=-spbk3k2*abb1(17)
      abb1(18)=abb1(18)+abb1(20)
      abb1(18)=2.0_ki*abb1(18)
      abb1(20)=spbl4k2*spak2l4
      abb1(23)=-abb1(20)-2.0_ki*es23
      abb1(23)=abb1(11)*abb1(23)
      abb1(9)=-abb1(19)*abb1(9)*spak1k2*abb1(1)
      abb1(9)=abb1(9)+abb1(23)
      abb1(9)=2.0_ki*abb1(9)
      abb1(23)=16.0_ki*abb1(11)
      abb1(22)=abb1(11)*abb1(22)
      abb1(12)=-spbk3k2*abb1(12)*abb1(19)
      abb1(12)=abb1(22)+abb1(12)
      abb1(12)=2.0_ki*abb1(12)
      abb1(19)=es23+abb1(20)
      abb1(19)=abb1(19)*abb1(21)
      abb1(17)=spbl4k1*abb1(17)
      abb1(17)=2.0_ki*abb1(19)+abb1(17)
      abb1(17)=2.0_ki*abb1(17)
      abb1(11)=-8.0_ki*abb1(11)
      abb1(19)=-16.0_ki*abb1(21)
      R2d1=abb1(16)
      rat2 = rat2 + R2d1
      if (debug_nlo_diagrams) then
          write (logfile,*) "<result name='r2' index='1' value='", &
          & R2d1, "'/>"
      end if
   end subroutine
end module p0_gg_gh_abbrevd1h0
