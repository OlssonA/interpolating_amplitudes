module     p0_gg_gh_abbrevd11h0
   use p0_gg_gh_config, only: ki
   use p0_gg_gh_kinematics, only: epstensor
   use p0_gg_gh_globalsh0
   implicit none
   private
   complex(ki), dimension(15), public :: abb11
   complex(ki), public :: R2d11
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
      abb11(1)=sqrt(mT**2)
      abb11(2)=sqrt2**(-1)
      abb11(3)=spbk2k1**(-1)
      abb11(4)=spak2k3**(-1)
      abb11(5)=spbk3k2**(-1)
      abb11(6)=spak2l4**(-1)
      abb11(7)=spbl4k2**(-1)
      abb11(8)=c1-c2
      abb11(9)=i_*e*gHT*abb11(4)*abb11(3)*abb11(2)
      abb11(10)=abb11(8)*abb11(9)*abb11(5)*abb11(1)
      abb11(11)=-abb11(10)*spak1l4*spbl4k3
      abb11(8)=-abb11(9)*abb11(8)
      abb11(9)=abb11(8)*spak1k2
      abb11(12)=-abb11(1)*abb11(9)
      abb11(13)=abb11(7)*abb11(6)*abb11(12)*mH**2
      abb11(13)=abb11(13)+abb11(11)
      abb11(13)=es12*abb11(13)
      abb11(9)=abb11(1)**3*abb11(9)
      abb11(9)=2.0_ki*abb11(9)+abb11(13)
      abb11(9)=2.0_ki*abb11(9)
      abb11(11)=4.0_ki*abb11(11)
      abb11(12)=4.0_ki*abb11(12)
      abb11(13)=spak1k2*abb11(10)
      abb11(13)=2.0_ki*abb11(13)
      abb11(14)=spbl4k3*abb11(13)
      abb11(8)=-2.0_ki*spak2l4*abb11(8)*abb11(1)
      abb11(15)=spbl4k2*spak2l4
      abb11(15)=-es12+abb11(15)
      abb11(15)=-2.0_ki*abb11(10)*abb11(15)
      abb11(13)=spbk3k1*abb11(13)
      abb11(10)=-8.0_ki*abb11(10)
      R2d11=0.0_ki
      rat2 = rat2 + R2d11
      if (debug_nlo_diagrams) then
          write (logfile,*) "<result name='r2' index='11' value='", &
          & R2d11, "'/>"
      end if
   end subroutine
end module p0_gg_gh_abbrevd11h0
