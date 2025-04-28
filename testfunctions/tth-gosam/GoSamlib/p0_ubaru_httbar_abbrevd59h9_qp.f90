module     p0_ubaru_httbar_abbrevd59h9_qp
   use p0_ubaru_httbar_config, only: ki => ki_qp
   use p0_ubaru_httbar_kinematics_qp, only: epstensor
   use p0_ubaru_httbar_globalsh9_qp
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
      use p0_ubaru_httbar_kinematics_qp
      use p0_ubaru_httbar_model_qp
      use p0_ubaru_httbar_color_qp, only: TR
      use p0_ubaru_httbar_globalsl1_qp, only: epspow
      implicit none
      abb59(1)=sqrt(mT**2)
      abb59(2)=NC**(-1)
      abb59(3)=es12**(-1)
      abb59(4)=spbl4k2**(-1)
      abb59(5)=spak2l5**(-1)
      abb59(6)=spak2l4**(-1)
      abb59(7)=spbl5k2**(-1)
      abb59(8)=spak2l3**(-1)
      abb59(9)=spbl3k2**(-1)
      abb59(10)=spbl4k2*spak2l4
      abb59(11)=spak2l3*spbl3k2
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
      abb59(25)=spbl5k2*abb59(24)*abb59(1)**3
      abb59(26)=abb59(1)*abb59(24)
      abb59(27)=spal3l4*spbl3k2
      abb59(28)=-spbl5l4*abb59(26)*abb59(27)
      abb59(12)=abb59(28)+abb59(25)+abb59(12)
      abb59(12)=spak1l4*abb59(12)
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
      abb59(41)=abb59(27)*spak1k2
      abb59(34)=abb59(34)*abb59(41)
      abb59(31)=abb59(31)*abb59(32)
      abb59(38)=-abb59(38)*abb59(39)
      abb59(31)=abb59(31)+abb59(38)
      abb59(38)=spak1l3*spbl3k2
      abb59(42)=abb59(4)*spbl5k2
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
      abb59(46)=spak2l4*abb59(43)
      abb59(22)=abb59(22)*abb59(15)
      abb59(18)=abb59(18)*mT
      abb59(18)=abb59(22)-abb59(18)
      abb59(22)=abb59(18)*abb59(4)*spbl5k2**2
      abb59(47)=abb59(22)*spak1l5
      abb59(20)=-abb59(20)*abb59(47)
      abb59(48)=spal4l5*spbl5k2
      abb59(49)=-abb59(21)*spak1k2*abb59(48)
      abb59(12)=abb59(49)+abb59(20)+abb59(46)+abb59(45)+abb59(31)+abb59(34)+abb&
      &59(12)
      abb59(12)=2.0_ki*abb59(12)
      abb59(20)=abb59(38)*abb59(18)
      abb59(31)=abb59(20)*abb59(42)
      abb59(34)=4.0_ki*abb59(31)
      abb59(38)=abb59(16)*spak1l4
      abb59(45)=abb59(38)*abb59(11)
      abb59(46)=4.0_ki*abb59(45)
      abb59(14)=abb59(19)-abb59(14)
      abb59(19)=mT**3
      abb59(14)=abb59(19)*abb59(14)
      abb59(44)=abb59(14)*abb59(44)
      abb59(43)=abb59(44)+abb59(43)
      abb59(43)=abb59(4)*abb59(43)
      abb59(21)=spak1l4*abb59(21)
      abb59(19)=-abb59(7)*abb59(24)*abb59(19)*abb59(5)**2
      abb59(24)=abb59(41)*abb59(19)
      abb59(21)=abb59(24)+3.0_ki*abb59(21)+abb59(43)
      abb59(21)=4.0_ki*abb59(21)
      abb59(24)=spak1l4*spbl5k2*abb59(26)
      abb59(41)=abb59(16)*abb59(41)
      abb59(43)=abb59(16)*spak1k2
      abb59(44)=abb59(43)*abb59(48)
      abb59(10)=-abb59(38)*abb59(10)
      abb59(10)=abb59(10)+abb59(44)-abb59(47)+abb59(45)+abb59(31)+abb59(24)+abb&
      &59(41)
      abb59(10)=2.0_ki*abb59(10)
      abb59(24)=4.0_ki*abb59(38)
      abb59(31)=mH**2*abb59(9)*abb59(8)
      abb59(22)=spak1k2*abb59(22)*abb59(31)
      abb59(41)=spbl5l3*abb59(18)*spak1l3
      abb59(44)=abb59(42)*abb59(41)
      abb59(22)=abb59(22)+abb59(44)
      abb59(22)=2.0_ki*abb59(22)
      abb59(16)=abb59(16)*spak1l5
      abb59(44)=abb59(11)*abb59(16)
      abb59(20)=abb59(20)+abb59(44)
      abb59(20)=2.0_ki*abb59(20)
      abb59(44)=2.0_ki*abb59(26)
      abb59(45)=-spak1l3*spbl5k2*abb59(44)
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
      abb59(13)=spbl5k2*abb59(13)*abb59(31)
      abb59(15)=3.0_ki*abb59(37)+2.0_ki*abb59(35)
      abb59(17)=abb59(15)*abb59(40)
      abb59(19)=abb59(11)*abb59(19)
      abb59(13)=abb59(19)+abb59(13)+abb59(30)+abb59(17)
      abb59(13)=spak1k2*abb59(13)
      abb59(17)=2.0_ki*spak1k2
      abb59(17)=-abb59(17)*abb59(23)
      abb59(19)=spak1k2*abb59(4)
      abb59(11)=abb59(14)*abb59(11)*abb59(19)
      abb59(11)=abb59(17)+abb59(11)
      abb59(11)=abb59(6)*abb59(11)
      abb59(14)=-spbl5l4*spak1l4*abb59(44)
      abb59(11)=abb59(41)+abb59(14)+abb59(11)+abb59(13)
      abb59(11)=2.0_ki*abb59(11)
      abb59(13)=2.0_ki*abb59(43)
      abb59(14)=2.0_ki*abb59(29)
      abb59(14)=spbl3k2*abb59(19)*abb59(14)
      abb59(17)=2.0_ki*abb59(27)
      abb59(16)=abb59(17)*abb59(16)
      abb59(17)=abb59(38)*abb59(17)
      abb59(19)=abb59(27)*abb59(44)
      abb59(23)=4.0_ki*abb59(26)
      abb59(25)=abb59(29)*spak2l4
      abb59(26)=-abb59(28)*abb59(32)
      abb59(15)=abb59(15)*abb59(39)
      abb59(15)=abb59(26)+abb59(15)
      abb59(15)=abb59(15)*abb59(42)
      abb59(15)=abb59(15)-2.0_ki*abb59(25)
      abb59(15)=2.0_ki*abb59(15)
      abb59(25)=-8.0_ki*abb59(29)*abb59(4)
      abb59(18)=-2.0_ki*abb59(18)*abb59(42)
      R2d59=0.0_ki
      rat2 = rat2 + R2d59
      if (debug_nlo_diagrams) then
          write (logfile,*) "<result name='r2' index='59' value='", &
          & R2d59, "'/>"
      end if
   end subroutine
end module p0_ubaru_httbar_abbrevd59h9_qp
