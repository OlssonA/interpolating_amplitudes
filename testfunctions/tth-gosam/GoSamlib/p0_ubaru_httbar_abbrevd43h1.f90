module     p0_ubaru_httbar_abbrevd43h1
   use p0_ubaru_httbar_config, only: ki
   use p0_ubaru_httbar_kinematics, only: epstensor
   use p0_ubaru_httbar_globalsh1
   implicit none
   private
   complex(ki), dimension(27), public :: abb43
   complex(ki), public :: R2d43
   public :: init_abbrev
   complex(ki), parameter :: i_ = (0.0_ki, 1.0_ki)
contains
   subroutine     init_abbrev()
      use p0_ubaru_httbar_config, only: deltaOS, &
     &    logfile, debug_nlo_diagrams
      use p0_ubaru_httbar_kinematics
      use p0_ubaru_httbar_model
      use p0_ubaru_httbar_color, only: TR
      use p0_ubaru_httbar_globalsl1, only: epspow
      implicit none
      abb43(1)=1.0_ki/(mH**2+mT**2-es34-es45+es12)
      abb43(2)=NC**(-1)
      abb43(3)=es12**(-1)
      abb43(4)=1.0_ki/(-mT**2+es34)
      abb43(5)=sqrt(mT**2)
      abb43(6)=spbl4k2**(-1)
      abb43(7)=spak2l3**(-1)
      abb43(8)=spbl3k2**(-1)
      abb43(9)=spbl5k2**(-1)
      abb43(10)=-spal3l5*spak1l4*abb43(1)
      abb43(11)=-spal3l4*spak1l5*abb43(4)
      abb43(10)=abb43(10)+abb43(11)
      abb43(11)=gHT*gs**4*i_*e
      abb43(12)=TR*abb43(3)
      abb43(12)=abb43(11)*abb43(12)**2
      abb43(13)=mT**2
      abb43(14)=abb43(13)*abb43(12)
      abb43(11)=abb43(3)*abb43(11)*TR**2
      abb43(11)=-1.0_ki/3.0_ki*abb43(11)+2.0_ki*abb43(14)
      abb43(14)=c1*abb43(2)
      abb43(14)=abb43(14)-c2
      abb43(10)=2.0_ki*spbl3k2*abb43(10)*abb43(14)*abb43(11)
      abb43(11)=abb43(4)*abb43(12)
      abb43(14)=abb43(11)*abb43(2)
      abb43(15)=abb43(14)*c1
      abb43(16)=abb43(11)*c2
      abb43(15)=abb43(15)-abb43(16)
      abb43(16)=abb43(15)*spal3l4
      abb43(17)=abb43(16)*spak1l5
      abb43(12)=abb43(12)*abb43(1)
      abb43(18)=abb43(2)*abb43(12)
      abb43(19)=abb43(18)*c1
      abb43(20)=abb43(12)*c2
      abb43(19)=abb43(19)-abb43(20)
      abb43(20)=abb43(19)*spal3l5
      abb43(21)=abb43(20)*spak1l4
      abb43(17)=abb43(17)+abb43(21)
      abb43(21)=spbl3k2*abb43(17)
      abb43(21)=4.0_ki*abb43(21)
      abb43(22)=-abb43(5)**2*abb43(21)
      abb43(23)=2.0_ki*spak1k2
      abb43(23)=abb43(23)*spbl3k2
      abb43(24)=-abb43(16)*abb43(23)
      abb43(23)=-abb43(20)*abb43(23)
      abb43(17)=2.0_ki*spbk2k1*abb43(17)
      abb43(14)=abb43(14)+abb43(18)
      abb43(14)=abb43(14)*c1
      abb43(11)=abb43(12)+abb43(11)
      abb43(11)=abb43(11)*c2
      abb43(11)=abb43(14)-abb43(11)
      abb43(12)=-abb43(5)*mT*abb43(11)
      abb43(11)=-abb43(13)*abb43(11)
      abb43(13)=abb43(12)+abb43(11)
      abb43(13)=abb43(6)*abb43(13)
      abb43(14)=-spak1l5*abb43(13)
      abb43(12)=abb43(9)*abb43(12)
      abb43(11)=abb43(11)*abb43(9)
      abb43(12)=abb43(12)+abb43(11)
      abb43(18)=-spak1l4*abb43(12)
      abb43(19)=abb43(19)*spak2l5
      abb43(25)=spak1l4*abb43(19)
      abb43(15)=abb43(15)*spak2l4
      abb43(26)=spak1l5*abb43(15)
      abb43(25)=abb43(26)+abb43(25)
      abb43(26)=abb43(7)*abb43(8)*mH**2
      abb43(25)=abb43(26)*abb43(25)
      abb43(11)=spbl3k2*abb43(11)*abb43(6)
      abb43(27)=-spak1l3*abb43(11)
      abb43(14)=abb43(14)+abb43(18)+abb43(27)+abb43(25)
      abb43(14)=2.0_ki*spbk2k1*abb43(14)
      abb43(16)=-4.0_ki*abb43(16)
      abb43(15)=-abb43(26)*abb43(15)
      abb43(13)=abb43(15)+abb43(13)
      abb43(13)=4.0_ki*abb43(13)
      abb43(15)=-4.0_ki*abb43(20)
      abb43(18)=-abb43(26)*abb43(19)
      abb43(12)=abb43(18)+abb43(12)
      abb43(12)=4.0_ki*abb43(12)
      abb43(11)=4.0_ki*abb43(11)
      R2d43=abb43(10)
      rat2 = rat2 + R2d43
      if (debug_nlo_diagrams) then
          write (logfile,*) "<result name='r2' index='43' value='", &
          & R2d43, "'/>"
      end if
   end subroutine
end module p0_ubaru_httbar_abbrevd43h1
