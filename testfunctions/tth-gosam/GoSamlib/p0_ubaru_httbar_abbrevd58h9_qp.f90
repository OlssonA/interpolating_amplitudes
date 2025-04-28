module     p0_ubaru_httbar_abbrevd58h9_qp
   use p0_ubaru_httbar_config, only: ki => ki_qp
   use p0_ubaru_httbar_kinematics_qp, only: epstensor
   use p0_ubaru_httbar_globalsh9_qp
   implicit none
   private
   complex(ki), dimension(46), public :: abb58
   complex(ki), public :: R2d58
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
      abb58(1)=sqrt(mT**2)
      abb58(2)=NC**(-1)
      abb58(3)=es12**(-1)
      abb58(4)=spbl4k2**(-1)
      abb58(5)=spak2l5**(-1)
      abb58(6)=spak2l3**(-1)
      abb58(7)=spbl3k2**(-1)
      abb58(8)=spbl5k2**(-1)
      abb58(9)=mH**2*abb58(7)*abb58(6)
      abb58(10)=abb58(9)*spak1k2
      abb58(11)=spbl5k2**2
      abb58(12)=spal4l5*abb58(11)*abb58(10)
      abb58(13)=spbl5l3*spak1l3
      abb58(14)=spal4l5*spbl5k2
      abb58(15)=abb58(14)*abb58(13)
      abb58(12)=abb58(15)+abb58(12)
      abb58(15)=i_*e*gHT*abb58(3)*TR**2*gs**4
      abb58(16)=abb58(15)*c2
      abb58(17)=abb58(16)*abb58(2)
      abb58(18)=abb58(15)*abb58(2)**2
      abb58(19)=abb58(18)*c1
      abb58(17)=abb58(17)-abb58(19)
      abb58(20)=abb58(1)*abb58(17)
      abb58(12)=abb58(20)*abb58(12)
      abb58(21)=spak1l4*spbl4k2*spak2l4
      abb58(22)=spak1k2*spbl3k2
      abb58(23)=abb58(22)*spal3l4
      abb58(24)=abb58(14)*spak1k2
      abb58(21)=-abb58(21)+abb58(23)+abb58(24)
      abb58(23)=mT*abb58(2)
      abb58(16)=abb58(16)*abb58(23)
      abb58(19)=abb58(19)*mT
      abb58(16)=abb58(19)-abb58(16)
      abb58(19)=abb58(5)*abb58(16)
      abb58(24)=abb58(1)**2
      abb58(25)=-abb58(24)*abb58(19)
      abb58(26)=abb58(25)*abb58(21)
      abb58(27)=abb58(15)*abb58(1)
      abb58(28)=abb58(27)*abb58(23)
      abb58(29)=mT**2
      abb58(30)=abb58(15)*abb58(29)*abb58(2)
      abb58(31)=abb58(30)+abb58(28)
      abb58(32)=c2*abb58(5)
      abb58(31)=abb58(31)*abb58(32)
      abb58(33)=abb58(29)*abb58(18)
      abb58(34)=abb58(18)*mT
      abb58(35)=abb58(34)*abb58(1)
      abb58(36)=-abb58(33)-abb58(35)
      abb58(37)=c1*abb58(5)
      abb58(36)=abb58(36)*abb58(37)
      abb58(31)=abb58(31)+abb58(36)
      abb58(36)=spak1l4*spbl3k2
      abb58(38)=abb58(36)*spak2l3
      abb58(31)=abb58(38)*abb58(1)*abb58(31)
      abb58(24)=abb58(24)*abb58(16)
      abb58(11)=abb58(11)*abb58(4)
      abb58(39)=-abb58(24)*abb58(11)
      abb58(40)=abb58(5)*abb58(17)
      abb58(29)=-abb58(40)*abb58(29)*abb58(1)
      abb58(41)=abb58(29)*spak2l3
      abb58(42)=abb58(4)*spbl5k2
      abb58(43)=abb58(42)*spbl3k2
      abb58(44)=-abb58(43)*abb58(41)
      abb58(39)=abb58(39)+abb58(44)
      abb58(39)=spak1l5*abb58(39)
      abb58(44)=spak1l4*spbl5k2
      abb58(45)=abb58(44)*abb58(17)*abb58(1)**3
      abb58(43)=abb58(43)*spak1l3
      abb58(46)=abb58(24)*abb58(43)
      abb58(12)=abb58(39)+abb58(31)+abb58(46)+abb58(45)+abb58(26)+abb58(12)
      abb58(12)=2.0_ki*abb58(12)
      abb58(26)=-2.0_ki*abb58(30)-3.0_ki*abb58(28)
      abb58(26)=abb58(26)*abb58(32)
      abb58(28)=2.0_ki*abb58(33)+3.0_ki*abb58(35)
      abb58(28)=abb58(28)*abb58(37)
      abb58(26)=abb58(26)+abb58(28)
      abb58(26)=spak1l4*abb58(1)*abb58(26)
      abb58(28)=spak1l5*abb58(29)*abb58(42)
      abb58(26)=abb58(26)+2.0_ki*abb58(28)
      abb58(26)=4.0_ki*abb58(26)
      abb58(28)=-spak1l5*abb58(11)
      abb58(28)=abb58(28)+abb58(43)
      abb58(28)=abb58(16)*abb58(28)
      abb58(21)=-abb58(38)-abb58(21)
      abb58(21)=abb58(19)*abb58(21)
      abb58(30)=abb58(20)*abb58(44)
      abb58(21)=abb58(30)+abb58(21)+abb58(28)
      abb58(21)=2.0_ki*abb58(21)
      abb58(28)=4.0_ki*spak1l4*abb58(19)
      abb58(16)=2.0_ki*abb58(16)
      abb58(11)=abb58(16)*abb58(11)
      abb58(30)=spak1l3*abb58(11)
      abb58(10)=abb58(11)*abb58(10)
      abb58(11)=spbl5k2*abb58(16)*spak1l3
      abb58(15)=abb58(23)*abb58(15)
      abb58(23)=abb58(27)*abb58(2)
      abb58(15)=abb58(23)-abb58(15)
      abb58(15)=abb58(15)*c2
      abb58(18)=abb58(18)*abb58(1)
      abb58(18)=abb58(18)-abb58(34)
      abb58(18)=abb58(18)*c1
      abb58(15)=abb58(15)-abb58(18)
      abb58(9)=spbl5k2*abb58(15)*abb58(9)
      abb58(9)=3.0_ki*abb58(25)+abb58(9)
      abb58(9)=spak1k2*abb58(9)
      abb58(13)=abb58(20)*abb58(13)
      abb58(9)=abb58(13)+abb58(9)
      abb58(9)=2.0_ki*abb58(9)
      abb58(13)=2.0_ki*abb58(19)
      abb58(18)=-spak1k2*abb58(13)
      abb58(15)=-abb58(15)*abb58(36)
      abb58(19)=spal4l5*abb58(19)*abb58(22)
      abb58(15)=abb58(15)+abb58(19)
      abb58(15)=2.0_ki*abb58(15)
      abb58(13)=-spak2l4*abb58(36)*abb58(13)
      abb58(19)=abb58(4)*abb58(40)
      abb58(17)=abb58(8)*spak2l4*abb58(17)*abb58(5)**2
      abb58(17)=abb58(19)+abb58(17)
      abb58(17)=2.0_ki*abb58(22)*abb58(17)*mT**3
      abb58(19)=4.0_ki*abb58(20)
      abb58(14)=abb58(14)*abb58(19)
      abb58(20)=abb58(24)*abb58(42)
      abb58(22)=-abb58(41)*abb58(4)*spbl3k2
      abb58(20)=-3.0_ki*abb58(20)+abb58(22)
      abb58(20)=2.0_ki*abb58(20)
      abb58(22)=8.0_ki*abb58(4)*abb58(29)
      abb58(16)=-abb58(42)*abb58(16)
      R2d58=0.0_ki
      rat2 = rat2 + R2d58
      if (debug_nlo_diagrams) then
          write (logfile,*) "<result name='r2' index='58' value='", &
          & R2d58, "'/>"
      end if
   end subroutine
end module p0_ubaru_httbar_abbrevd58h9_qp
