module     p2_gg_httbar_abbrevd7h8
   use p2_gg_httbar_config, only: ki
   use p2_gg_httbar_kinematics, only: epstensor
   use p2_gg_httbar_globalsh8
   implicit none
   private
   complex(ki), dimension(9), public :: abb7
   complex(ki), public :: R2d7
   public :: init_abbrev
   complex(ki), parameter :: i_ = (0.0_ki, 1.0_ki)
contains
   subroutine     init_abbrev()
      use p2_gg_httbar_config, only: deltaOS, &
     &    logfile, debug_nlo_diagrams
      use p2_gg_httbar_kinematics
      use p2_gg_httbar_model
      use p2_gg_httbar_color, only: TR
      use p2_gg_httbar_globalsl1, only: epspow
      implicit none
      abb7(1)=1.0_ki/(mH**2+mT**2-es34-es45+es12)
      abb7(2)=es12**(-1)
      abb7(3)=spak2l5**(-1)
      abb7(4)=sqrt(mT**2)
      abb7(5)=1.0_ki/(-mT**2+es34)
      abb7(6)=spbl4k2**(-1)
      abb7(7)=c2-c1
      abb7(8)=abb7(1)+abb7(5)
      abb7(9)=-abb7(4)*abb7(7)*abb7(8)
      abb7(7)=abb7(7)*mT
      abb7(8)=-abb7(8)*abb7(7)
      abb7(8)=abb7(8)+abb7(9)
      abb7(8)=spak2l4*abb7(8)
      abb7(7)=abb7(7)*spbl3k2*spak2l3
      abb7(9)=-abb7(5)*abb7(6)*abb7(7)
      abb7(8)=abb7(9)+abb7(8)
      abb7(8)=spbl5k2*abb7(8)
      abb7(7)=-spak2l4*abb7(1)*abb7(3)*abb7(7)
      abb7(7)=abb7(7)+abb7(8)
      abb7(7)=9.0_ki/8.0_ki*abb7(2)*gHT*e*spbe2e1*spae1e2*NC*TR*i_*abb7(7)*gs**4
      R2d7=0.0_ki
      rat2 = rat2 + R2d7
      if (debug_nlo_diagrams) then
          write (logfile,*) "<result name='r2' index='7' value='", &
          & R2d7, "'/>"
      end if
   end subroutine
end module p2_gg_httbar_abbrevd7h8
