module     p0_ubaru_httbar_abbrevd59h6
   use p0_ubaru_httbar_config, only: ki
   use p0_ubaru_httbar_kinematics, only: epstensor
   use p0_ubaru_httbar_globalsh6
   implicit none
   private
   complex(ki), dimension(49), public :: abb59
   complex(ki), public :: R2d59
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
      abb59(1)=sqrt(mT**2)
      abb59(2)=NC**(-1)
      abb59(3)=es12**(-1)
      abb59(4)=spak2l4**(-1)
      abb59(5)=spbl5k2**(-1)
      abb59(6)=spbl4k2**(-1)
      abb59(7)=spak2l5**(-1)
      abb59(8)=spak2l3**(-1)
      abb59(9)=spbl3k2**(-1)
      abb59(10)=spak2l4*spbl4k2
      abb59(11)=spbl3k2*spak2l3
      abb59(12)=abb59(10)-abb59(11)
      abb59(13)=i_*e*gHT*abb59(3)*TR**2*gs**4
      abb59(14)=c2*abb59(13)*abb59(5)
      abb59(15)=mT*abb59(2)
      abb59(16)=abb59(15)*abb59(14)
      abb59(17)=abb59(13)*abb59(2)**2
      abb59(18)=abb59(17)*c1
      abb59(19)=abb59(18)*abb59(5)
      abb59(20)=mT*abb59(19)
      abb59(16)=abb59(20)-abb59(16)
      abb59(20)=abb59(1)**2
      abb59(21)=-abb59(20)*abb59(16)
      abb59(12)=abb59(21)*abb59(12)
      abb59(22)=abb59(13)*c2
      abb59(23)=abb59(22)*abb59(2)
      abb59(24)=abb59(23)-abb59(18)
      abb59(25)=spak2l5*abb59(24)*abb59(1)**3
      abb59(26)=abb59(1)*abb59(24)
      abb59(27)=spbl4l3*spak2l3
      abb59(28)=-spal4l5*abb59(26)*abb59(27)
      abb59(12)=abb59(28)+abb59(25)+abb59(12)
      abb59(12)=spbl4k1*abb59(12)
      abb59(25)=abb59(13)*abb59(1)
      abb59(28)=abb59(25)*abb59(15)
      abb59(29)=mT**2
      abb59(30)=abb59(13)*abb59(29)*abb59(2)
      abb59(31)=abb59(28)+abb59(30)
      abb59(32)=c2*abb59(1)
      abb59(33)=abb59(32)*abb59(5)
      abb59(34)=-abb59(31)*abb59(33)
      abb59(35)=abb59(29)*abb59(17)
      abb59(36)=abb59(17)*mT
      abb59(37)=abb59(36)*abb59(1)
      abb59(38)=abb59(37)+abb59(35)
      abb59(39)=c1*abb59(1)
      abb59(40)=abb59(5)*abb59(39)
      abb59(41)=abb59(38)*abb59(40)
      abb59(34)=abb59(34)+abb59(41)
      abb59(41)=abb59(27)*spbk2k1
      abb59(34)=abb59(34)*abb59(41)
      abb59(31)=abb59(31)*abb59(32)
      abb59(38)=-abb59(38)*abb59(39)
      abb59(31)=abb59(31)+abb59(38)
      abb59(38)=spbl3k1*spak2l3
      abb59(42)=abb59(4)*spak2l5
      abb59(31)=abb59(31)*abb59(38)*abb59(42)
      abb59(29)=abb59(29)*abb59(1)
      abb59(23)=abb59(23)*abb59(29)
      abb59(43)=abb59(29)*abb59(18)
      abb59(23)=abb59(23)-abb59(43)
      abb59(23)=abb59(23)*abb59(42)
      abb59(44)=abb59(41)*abb59(6)
      abb59(45)=-abb59(44)*abb59(23)
      abb59(14)=abb59(2)*abb59(14)
      abb59(29)=abb59(14)*abb59(29)
      abb59(43)=abb59(43)*abb59(5)
      abb59(29)=abb59(29)-abb59(43)
      abb59(43)=abb59(29)*abb59(38)
      abb59(46)=spbl4k2*abb59(43)
      abb59(22)=abb59(22)*abb59(15)
      abb59(18)=abb59(18)*mT
      abb59(18)=abb59(22)-abb59(18)
      abb59(22)=abb59(18)*abb59(4)*spak2l5**2
      abb59(47)=abb59(22)*spbl5k1
      abb59(20)=-abb59(20)*abb59(47)
      abb59(48)=spbl5l4*spak2l5
      abb59(49)=-abb59(21)*spbk2k1*abb59(48)
      abb59(12)=abb59(49)+abb59(20)+abb59(46)+abb59(45)+abb59(31)+abb59(34)+abb&
      &59(12)
      abb59(12)=2.0_ki*abb59(12)
      abb59(20)=abb59(38)*abb59(18)
      abb59(31)=abb59(20)*abb59(42)
      abb59(34)=4.0_ki*abb59(31)
      abb59(38)=abb59(16)*spbl4k1
      abb59(45)=abb59(38)*abb59(11)
      abb59(46)=4.0_ki*abb59(45)
      abb59(14)=abb59(19)-abb59(14)
      abb59(19)=mT**3
      abb59(14)=abb59(19)*abb59(14)
      abb59(44)=abb59(14)*abb59(44)
      abb59(43)=abb59(44)+abb59(43)
      abb59(43)=abb59(4)*abb59(43)
      abb59(21)=spbl4k1*abb59(21)
      abb59(19)=-abb59(7)*abb59(24)*abb59(19)*abb59(5)**2
      abb59(24)=abb59(41)*abb59(19)
      abb59(21)=abb59(24)+3.0_ki*abb59(21)+abb59(43)
      abb59(21)=4.0_ki*abb59(21)
      abb59(24)=spbl4k1*spak2l5*abb59(26)
      abb59(41)=abb59(16)*abb59(41)
      abb59(43)=abb59(16)*spbk2k1
      abb59(44)=abb59(43)*abb59(48)
      abb59(10)=-abb59(38)*abb59(10)
      abb59(10)=abb59(10)+abb59(44)-abb59(47)+abb59(45)+abb59(31)+abb59(24)+abb&
      &59(41)
      abb59(10)=2.0_ki*abb59(10)
      abb59(24)=4.0_ki*abb59(38)
      abb59(16)=abb59(16)*spbl5k1
      abb59(31)=abb59(11)*abb59(16)
      abb59(20)=abb59(20)+abb59(31)
      abb59(20)=2.0_ki*abb59(20)
      abb59(31)=2.0_ki*abb59(27)
      abb59(16)=abb59(31)*abb59(16)
      abb59(41)=2.0_ki*abb59(26)
      abb59(27)=abb59(27)*abb59(41)
      abb59(31)=abb59(38)*abb59(31)
      abb59(38)=-spbl3k1*spak2l5*abb59(41)
      abb59(44)=mH**2*abb59(9)*abb59(8)
      abb59(22)=spbk2k1*abb59(22)*abb59(44)
      abb59(45)=spal3l5*abb59(18)*spbl3k1
      abb59(47)=abb59(42)*abb59(45)
      abb59(22)=abb59(22)+abb59(47)
      abb59(22)=2.0_ki*abb59(22)
      abb59(28)=3.0_ki*abb59(28)+2.0_ki*abb59(30)
      abb59(30)=-abb59(28)*abb59(33)
      abb59(13)=abb59(15)*abb59(13)
      abb59(15)=-abb59(2)*abb59(25)
      abb59(13)=abb59(13)+abb59(15)
      abb59(13)=c2*abb59(13)
      abb59(15)=abb59(17)*abb59(1)
      abb59(15)=-abb59(36)+abb59(15)
      abb59(15)=c1*abb59(15)
      abb59(13)=abb59(13)+abb59(15)
      abb59(13)=spak2l5*abb59(13)*abb59(44)
      abb59(15)=3.0_ki*abb59(37)+2.0_ki*abb59(35)
      abb59(17)=abb59(15)*abb59(40)
      abb59(19)=abb59(11)*abb59(19)
      abb59(13)=abb59(19)+abb59(13)+abb59(30)+abb59(17)
      abb59(13)=spbk2k1*abb59(13)
      abb59(17)=2.0_ki*spbk2k1
      abb59(17)=-abb59(17)*abb59(23)
      abb59(19)=spbk2k1*abb59(4)
      abb59(11)=abb59(14)*abb59(11)*abb59(19)
      abb59(11)=abb59(17)+abb59(11)
      abb59(11)=abb59(6)*abb59(11)
      abb59(14)=-spal4l5*spbl4k1*abb59(41)
      abb59(11)=abb59(45)+abb59(14)+abb59(11)+abb59(13)
      abb59(11)=2.0_ki*abb59(11)
      abb59(13)=2.0_ki*abb59(43)
      abb59(14)=4.0_ki*abb59(26)
      abb59(17)=2.0_ki*abb59(29)
      abb59(17)=spak2l3*abb59(19)*abb59(17)
      abb59(19)=abb59(29)*spbl4k2
      abb59(23)=-abb59(28)*abb59(32)
      abb59(15)=abb59(15)*abb59(39)
      abb59(15)=abb59(23)+abb59(15)
      abb59(15)=abb59(15)*abb59(42)
      abb59(15)=abb59(15)-2.0_ki*abb59(19)
      abb59(15)=2.0_ki*abb59(15)
      abb59(19)=-8.0_ki*abb59(29)*abb59(4)
      abb59(18)=-2.0_ki*abb59(18)*abb59(42)
      R2d59=0.0_ki
      rat2 = rat2 + R2d59
      if (debug_nlo_diagrams) then
          write (logfile,*) "<result name='r2' index='59' value='", &
          & R2d59, "'/>"
      end if
   end subroutine
end module p0_ubaru_httbar_abbrevd59h6
