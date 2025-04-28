module     p0_ubaru_httbar_abbrevd71h2_qp
   use p0_ubaru_httbar_config, only: ki => ki_qp
   use p0_ubaru_httbar_kinematics_qp, only: epstensor
   use p0_ubaru_httbar_globalsh2_qp
   implicit none
   private
   complex(ki), dimension(51), public :: abb71
   complex(ki), public :: R2d71
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
      abb71(1)=1.0_ki/(mH**2+mT**2-es34-es45+es12)
      abb71(2)=NC**(-1)
      abb71(3)=es12**(-1)
      abb71(4)=spak2l3**(-1)
      abb71(5)=spbl3k2**(-1)
      abb71(6)=spbl4k2**(-1)
      abb71(7)=spbl5k2**(-1)
      abb71(8)=sqrt(mT**2)
      abb71(9)=spak2l5**(-1)
      abb71(10)=i_*e*gHT*abb71(3)*TR**2*gs**4
      abb71(11)=abb71(10)*spbk2k1*abb71(1)
      abb71(12)=abb71(11)*spak2l4
      abb71(13)=abb71(8)*mT
      abb71(14)=abb71(12)*abb71(13)
      abb71(15)=mT**2
      abb71(16)=abb71(12)*abb71(15)
      abb71(17)=abb71(14)+abb71(16)
      abb71(18)=abb71(7)*c2
      abb71(19)=abb71(18)*abb71(17)
      abb71(20)=abb71(11)*abb71(13)
      abb71(15)=abb71(15)*abb71(11)
      abb71(21)=abb71(20)+abb71(15)
      abb71(22)=spak2l5*abb71(6)
      abb71(23)=abb71(22)*c2
      abb71(24)=abb71(23)*abb71(21)
      abb71(19)=abb71(19)+abb71(24)
      abb71(19)=abb71(19)*NC
      abb71(25)=abb71(2)**2
      abb71(25)=abb71(25)-1.0_ki
      abb71(26)=abb71(25)*c1
      abb71(27)=c2*abb71(2)
      abb71(28)=abb71(26)-abb71(27)
      abb71(29)=abb71(15)*abb71(28)
      abb71(30)=abb71(6)*abb71(7)
      abb71(31)=abb71(29)*abb71(30)
      abb71(32)=abb71(30)*abb71(15)
      abb71(33)=NC*c2
      abb71(34)=abb71(32)*abb71(33)
      abb71(34)=abb71(31)+abb71(34)
      abb71(35)=spak2l3*spbl3k2
      abb71(34)=abb71(34)*abb71(35)
      abb71(36)=abb71(2)-NC
      abb71(10)=abb71(10)*spak2l4*abb71(1)
      abb71(36)=abb71(36)*c2*abb71(10)
      abb71(37)=c1*abb71(25)*abb71(10)
      abb71(36)=abb71(36)-abb71(37)
      abb71(37)=spbl3k1*spal3l5
      abb71(38)=-abb71(36)*abb71(37)
      abb71(34)=abb71(34)+abb71(38)
      abb71(38)=-abb71(7)*abb71(17)*abb71(28)
      abb71(25)=abb71(12)*abb71(25)
      abb71(39)=abb71(25)*c1
      abb71(27)=abb71(27)*abb71(12)
      abb71(39)=abb71(39)-abb71(27)
      abb71(40)=mH**2
      abb71(41)=abb71(40)*abb71(5)
      abb71(42)=abb71(41)*spak2l5
      abb71(43)=abb71(39)*abb71(42)
      abb71(12)=abb71(33)*abb71(12)
      abb71(44)=abb71(12)*abb71(42)
      abb71(43)=abb71(43)+abb71(44)
      abb71(43)=abb71(43)*abb71(4)
      abb71(45)=abb71(22)*abb71(21)*abb71(28)
      abb71(19)=abb71(34)+abb71(45)-abb71(38)+abb71(19)+abb71(43)
      abb71(38)=2.0_ki*abb71(19)
      abb71(43)=abb71(8)**2
      abb71(46)=-abb71(42)*abb71(43)*abb71(39)
      abb71(47)=abb71(20)*abb71(28)
      abb71(48)=abb71(33)*abb71(20)
      abb71(49)=abb71(47)+abb71(48)
      abb71(49)=abb71(6)*abb71(49)
      abb71(50)=spbl5k2*abb71(41)*abb71(49)*spak2l5**2
      abb71(44)=-abb71(43)*abb71(44)
      abb71(44)=abb71(50)+abb71(46)+abb71(44)
      abb71(44)=abb71(4)*abb71(44)
      abb71(46)=abb71(47)*abb71(22)
      abb71(47)=abb71(48)*abb71(22)
      abb71(47)=abb71(47)+abb71(46)
      abb71(48)=spbl5l3*abb71(47)
      abb71(10)=abb71(13)*abb71(10)
      abb71(50)=abb71(7)*abb71(10)*abb71(28)
      abb71(33)=abb71(33)*abb71(7)
      abb71(10)=abb71(33)*abb71(10)
      abb71(10)=abb71(50)+abb71(10)
      abb71(50)=abb71(10)*spbl5k1
      abb71(51)=-spbl3k2*abb71(50)
      abb71(48)=abb71(51)+abb71(48)
      abb71(48)=spal3l5*abb71(48)
      abb71(27)=-abb71(12)+abb71(27)
      abb71(51)=abb71(8)*mT**3
      abb71(27)=abb71(51)*abb71(27)
      abb71(25)=-c1*abb71(51)*abb71(25)
      abb71(25)=abb71(25)+abb71(27)
      abb71(25)=abb71(9)*abb71(25)*abb71(7)**2
      abb71(11)=abb71(11)*abb71(13)**2
      abb71(13)=abb71(26)*abb71(11)
      abb71(11)=c2*abb71(11)
      abb71(26)=abb71(11)*abb71(2)
      abb71(13)=abb71(13)-abb71(26)
      abb71(26)=-NC*abb71(11)
      abb71(26)=-abb71(13)+abb71(26)
      abb71(26)=abb71(30)*abb71(26)
      abb71(26)=-abb71(25)+abb71(26)
      abb71(26)=abb71(35)*abb71(26)
      abb71(21)=-abb71(22)*abb71(28)*abb71(21)*abb71(43)
      abb71(17)=abb71(17)*abb71(43)
      abb71(27)=-abb71(7)*abb71(17)*abb71(28)
      abb71(21)=abb71(27)+abb71(21)
      abb71(27)=abb71(7)*abb71(14)*abb71(28)
      abb71(27)=abb71(27)+abb71(46)
      abb71(27)=abb71(27)*abb71(40)
      abb71(17)=-abb71(18)*abb71(17)
      abb71(35)=-abb71(43)*abb71(24)
      abb71(17)=abb71(17)+abb71(35)
      abb71(35)=abb71(14)*abb71(18)
      abb71(20)=abb71(20)*abb71(23)
      abb71(20)=abb71(35)+abb71(20)
      abb71(20)=abb71(20)*abb71(40)
      abb71(17)=2.0_ki*abb71(17)+abb71(20)
      abb71(17)=NC*abb71(17)
      abb71(20)=abb71(43)*abb71(36)
      abb71(23)=abb71(20)*abb71(37)
      abb71(17)=abb71(23)+abb71(17)+2.0_ki*abb71(21)+abb71(27)+abb71(26)+abb71(&
      &48)+abb71(44)
      abb71(17)=2.0_ki*abb71(17)
      abb71(21)=8.0_ki*abb71(47)
      abb71(23)=-4.0_ki*abb71(34)
      abb71(14)=abb71(14)-abb71(16)
      abb71(18)=abb71(14)*abb71(18)
      abb71(18)=abb71(18)-abb71(24)
      abb71(18)=NC*abb71(18)
      abb71(14)=abb71(7)*abb71(14)*abb71(28)
      abb71(14)=abb71(18)+abb71(14)-abb71(45)
      abb71(14)=abb71(41)*abb71(14)
      abb71(12)=abb71(39)+abb71(12)
      abb71(18)=-abb71(4)*spak2l5*abb71(12)*abb71(5)**2*mH**4
      abb71(14)=abb71(18)+abb71(14)
      abb71(14)=abb71(4)*abb71(14)
      abb71(18)=2.0_ki*abb71(30)
      abb71(13)=abb71(13)*abb71(18)
      abb71(24)=-abb71(40)*abb71(31)
      abb71(11)=abb71(11)*abb71(18)
      abb71(18)=-c2*abb71(40)*abb71(32)
      abb71(11)=abb71(11)+abb71(18)
      abb71(11)=NC*abb71(11)
      abb71(11)=-2.0_ki*abb71(25)+abb71(14)+abb71(11)+abb71(13)+abb71(24)
      abb71(11)=4.0_ki*abb71(11)
      abb71(13)=-4.0_ki*abb71(19)
      abb71(14)=-abb71(7)*abb71(16)*abb71(28)
      abb71(16)=abb71(33)*abb71(16)
      abb71(14)=abb71(14)-abb71(16)
      abb71(14)=abb71(14)*abb71(9)
      abb71(16)=abb71(14)+abb71(49)
      abb71(16)=spak2l3*abb71(16)
      abb71(12)=abb71(12)*abb71(41)
      abb71(16)=-abb71(12)+abb71(16)
      abb71(16)=2.0_ki*abb71(16)
      abb71(18)=-spbl3k1*abb71(41)*abb71(36)
      abb71(18)=abb71(18)+2.0_ki*abb71(50)
      abb71(18)=2.0_ki*abb71(18)
      abb71(19)=4.0_ki*abb71(20)
      abb71(20)=abb71(29)*abb71(7)
      abb71(15)=abb71(33)*abb71(15)
      abb71(15)=abb71(15)+abb71(20)
      abb71(15)=abb71(15)*abb71(22)
      abb71(20)=2.0_ki*spbl3k2
      abb71(22)=-abb71(15)*abb71(20)
      abb71(24)=abb71(4)*abb71(42)*abb71(36)
      abb71(24)=abb71(24)+abb71(10)
      abb71(24)=spbl3k1*abb71(24)
      abb71(15)=-spbl5l3*abb71(15)
      abb71(15)=abb71(24)+abb71(15)
      abb71(15)=2.0_ki*abb71(15)
      abb71(10)=abb71(10)*abb71(20)
      abb71(12)=-abb71(4)*abb71(12)
      abb71(12)=abb71(14)+abb71(12)-abb71(49)
      abb71(12)=2.0_ki*spal3l5*abb71(12)
      R2d71=abb71(38)
      rat2 = rat2 + R2d71
      if (debug_nlo_diagrams) then
          write (logfile,*) "<result name='r2' index='71' value='", &
          & R2d71, "'/>"
      end if
   end subroutine
end module p0_ubaru_httbar_abbrevd71h2_qp
