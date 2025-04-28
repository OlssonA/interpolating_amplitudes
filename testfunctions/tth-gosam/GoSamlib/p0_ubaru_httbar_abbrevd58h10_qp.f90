module     p0_ubaru_httbar_abbrevd58h10_qp
   use p0_ubaru_httbar_config, only: ki => ki_qp
   use p0_ubaru_httbar_kinematics_qp, only: epstensor
   use p0_ubaru_httbar_globalsh10_qp
   implicit none
   private
   complex(ki), dimension(49), public :: abb58
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
      abb58(6)=spbl5k2**(-1)
      abb58(7)=spak2l3**(-1)
      abb58(8)=spbl3k2**(-1)
      abb58(9)=spak2l4**(-1)
      abb58(10)=i_*e*gHT*abb58(3)*TR**2*gs**4
      abb58(11)=abb58(10)*abb58(1)
      abb58(12)=mT*abb58(2)
      abb58(13)=abb58(11)*abb58(12)
      abb58(14)=mT**2
      abb58(15)=abb58(10)*abb58(2)
      abb58(16)=abb58(14)*abb58(15)
      abb58(17)=abb58(13)+abb58(16)
      abb58(18)=c2*abb58(17)
      abb58(19)=abb58(2)**2
      abb58(20)=abb58(11)*abb58(19)
      abb58(21)=mT*abb58(20)
      abb58(22)=abb58(12)**2
      abb58(23)=abb58(22)*abb58(10)
      abb58(24)=abb58(21)+abb58(23)
      abb58(25)=-c1*abb58(24)
      abb58(18)=abb58(18)+abb58(25)
      abb58(25)=abb58(5)*spak2l4
      abb58(26)=abb58(25)*abb58(1)
      abb58(18)=abb58(18)*abb58(26)
      abb58(22)=abb58(22)*abb58(11)
      abb58(27)=c1*abb58(4)
      abb58(28)=abb58(22)*abb58(27)
      abb58(11)=abb58(11)*abb58(2)
      abb58(29)=abb58(11)*c2
      abb58(30)=abb58(29)*abb58(14)
      abb58(31)=abb58(30)*abb58(4)
      abb58(28)=abb58(28)-abb58(31)
      abb58(31)=abb58(28)*spbl5k2
      abb58(18)=-abb58(31)+abb58(18)
      abb58(32)=spbl3k1*spak2l3
      abb58(18)=abb58(32)*abb58(18)
      abb58(33)=c2*abb58(4)
      abb58(17)=-abb58(17)*abb58(33)
      abb58(24)=abb58(24)*abb58(27)
      abb58(17)=abb58(17)+abb58(24)
      abb58(17)=abb58(1)*abb58(17)
      abb58(24)=abb58(22)*c1
      abb58(24)=abb58(24)-abb58(30)
      abb58(24)=abb58(24)*abb58(25)*abb58(6)
      abb58(17)=abb58(24)+abb58(17)
      abb58(30)=spbk2k1*spak2l3
      abb58(34)=abb58(30)*spbl5l3
      abb58(17)=abb58(34)*abb58(17)
      abb58(35)=spbl5k1*spak2l5*spbl5k2
      abb58(36)=spbl5k1*spak2l3
      abb58(37)=abb58(36)*spbl3k2
      abb58(38)=spbk2k1*spak2l4
      abb58(39)=abb58(38)*spbl5l4
      abb58(35)=-abb58(35)+abb58(37)-abb58(39)
      abb58(19)=abb58(19)*abb58(10)
      abb58(37)=abb58(19)*c1
      abb58(39)=abb58(37)*mT
      abb58(40)=abb58(10)*c2
      abb58(41)=abb58(40)*abb58(12)
      abb58(39)=abb58(39)-abb58(41)
      abb58(41)=-abb58(4)*abb58(39)
      abb58(42)=abb58(1)**2
      abb58(43)=-abb58(42)*abb58(41)
      abb58(44)=abb58(43)*abb58(35)
      abb58(45)=abb58(39)*abb58(5)*spak2l4**2
      abb58(46)=abb58(45)*spbl4k1
      abb58(42)=abb58(42)*abb58(46)
      abb58(40)=abb58(40)*abb58(2)
      abb58(37)=abb58(40)-abb58(37)
      abb58(40)=spbl5k1*spak2l4
      abb58(47)=abb58(37)*abb58(1)**3*abb58(40)
      abb58(48)=c1*abb58(20)
      abb58(29)=abb58(48)-abb58(29)
      abb58(48)=abb58(29)*spal4l5
      abb58(49)=-spbl5l3*abb58(36)*abb58(48)
      abb58(17)=abb58(42)+abb58(49)+abb58(47)+abb58(44)+abb58(17)+abb58(18)
      abb58(17)=2.0_ki*abb58(17)
      abb58(18)=spbl5k1*abb58(43)
      abb58(28)=abb58(28)*abb58(5)
      abb58(42)=-abb58(32)*abb58(28)
      abb58(18)=-3.0_ki*abb58(18)+abb58(42)
      abb58(18)=4.0_ki*abb58(18)
      abb58(34)=-abb58(34)-abb58(35)
      abb58(34)=abb58(41)*abb58(34)
      abb58(32)=-abb58(39)*abb58(25)*abb58(32)
      abb58(35)=-abb58(29)*abb58(40)
      abb58(32)=abb58(46)+abb58(35)+abb58(32)+abb58(34)
      abb58(32)=2.0_ki*abb58(32)
      abb58(34)=4.0_ki*spbl5k1*abb58(41)
      abb58(35)=2.0_ki*spbl5k2
      abb58(35)=-abb58(41)*abb58(36)*abb58(35)
      abb58(36)=abb58(39)*abb58(36)
      abb58(40)=-spbl5l4*abb58(41)*abb58(30)
      abb58(36)=abb58(36)+abb58(40)
      abb58(36)=2.0_ki*abb58(36)
      abb58(40)=-2.0_ki*spak2l3*abb58(29)*spbl5l3
      abb58(42)=abb58(19)*mT
      abb58(20)=abb58(20)-abb58(42)
      abb58(20)=abb58(20)*c1
      abb58(10)=abb58(12)*abb58(10)
      abb58(10)=abb58(11)-abb58(10)
      abb58(10)=abb58(10)*c2
      abb58(10)=abb58(20)-abb58(10)
      abb58(12)=2.0_ki*spbl3k1
      abb58(20)=spak2l4*abb58(10)*abb58(12)
      abb58(12)=-abb58(45)*abb58(12)
      abb58(13)=3.0_ki*abb58(13)+2.0_ki*abb58(16)
      abb58(16)=-abb58(13)*abb58(33)
      abb58(21)=3.0_ki*abb58(21)+2.0_ki*abb58(23)
      abb58(23)=abb58(21)*abb58(27)
      abb58(16)=abb58(16)+abb58(23)
      abb58(16)=spbk2k1*abb58(1)*abb58(16)
      abb58(23)=abb58(7)*abb58(8)*mH**2
      abb58(10)=abb58(10)*abb58(38)*abb58(23)
      abb58(38)=2.0_ki*spbk2k1
      abb58(24)=abb58(38)*abb58(24)
      abb58(42)=spbl5k1*abb58(48)
      abb58(10)=-2.0_ki*abb58(42)+abb58(24)+abb58(16)+abb58(10)
      abb58(10)=2.0_ki*abb58(10)
      abb58(16)=-abb58(41)*abb58(38)
      abb58(24)=-4.0_ki*abb58(29)
      abb58(23)=-abb58(45)*abb58(38)*abb58(23)
      abb58(29)=mT**3
      abb58(15)=-abb58(29)*abb58(15)
      abb58(11)=abb58(14)*abb58(11)
      abb58(11)=abb58(15)+abb58(11)
      abb58(11)=abb58(11)*abb58(33)
      abb58(14)=abb58(29)*abb58(19)
      abb58(14)=abb58(14)-abb58(22)
      abb58(14)=abb58(14)*abb58(27)
      abb58(11)=abb58(11)+abb58(14)
      abb58(11)=abb58(5)*abb58(11)
      abb58(14)=-abb58(9)*spbl5k2*abb58(29)*abb58(37)*abb58(4)**2
      abb58(11)=abb58(11)+abb58(14)
      abb58(11)=2.0_ki*abb58(30)*abb58(11)
      abb58(13)=-c2*abb58(13)
      abb58(14)=c1*abb58(21)
      abb58(13)=abb58(13)+abb58(14)
      abb58(13)=abb58(13)*abb58(26)
      abb58(13)=abb58(13)+2.0_ki*abb58(31)
      abb58(13)=2.0_ki*abb58(13)
      abb58(14)=8.0_ki*abb58(28)
      abb58(15)=2.0_ki*abb58(39)
      abb58(15)=abb58(25)*abb58(15)
      R2d58=0.0_ki
      rat2 = rat2 + R2d58
      if (debug_nlo_diagrams) then
          write (logfile,*) "<result name='r2' index='58' value='", &
          & R2d58, "'/>"
      end if
   end subroutine
end module p0_ubaru_httbar_abbrevd58h10_qp
