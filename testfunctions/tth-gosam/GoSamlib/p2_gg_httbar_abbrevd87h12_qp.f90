module     p2_gg_httbar_abbrevd87h12_qp
   use p2_gg_httbar_config, only: ki => ki_qp
   use p2_gg_httbar_kinematics_qp, only: epstensor
   use p2_gg_httbar_globalsh12_qp
   implicit none
   private
   complex(ki), dimension(52), public :: abb87
   complex(ki), public :: R2d87
   public :: init_abbrev
   complex(ki), parameter :: i_ = (0.0_ki, 1.0_ki)
contains
   subroutine     init_abbrev()
      use p2_gg_httbar_config, only: deltaOS, &
     &    logfile, debug_nlo_diagrams
      use p2_gg_httbar_kinematics_qp
      use p2_gg_httbar_model_qp
      use p2_gg_httbar_color_qp, only: TR
      use p2_gg_httbar_globalsl1_qp, only: epspow
      implicit none
      abb87(1)=1.0_ki/(mH**2+mT**2-es34-es45+es12)
      abb87(2)=sqrt(mT**2)
      abb87(3)=NC**(-1)
      abb87(4)=spak2l4**(-1)
      abb87(5)=spak2l5**(-1)
      abb87(6)=spak2l3**(-1)
      abb87(7)=spbl3k2**(-1)
      abb87(8)=mT*abb87(2)
      abb87(9)=i_*TR*e*gHT*abb87(1)*gs**4
      abb87(10)=abb87(8)*abb87(9)
      abb87(11)=c1*abb87(3)
      abb87(11)=abb87(11)-c3
      abb87(12)=-abb87(10)*abb87(11)
      abb87(13)=spak2l3*abb87(5)
      abb87(14)=abb87(12)*abb87(13)
      abb87(15)=abb87(14)*spbl3k2
      abb87(16)=spak1k2*abb87(15)
      abb87(17)=abb87(9)*abb87(2)**2
      abb87(18)=-abb87(17)*abb87(11)
      abb87(19)=abb87(18)*spbl5l3
      abb87(20)=-spak1l3*abb87(19)
      abb87(16)=abb87(20)+abb87(16)
      abb87(20)=spbl4e2*spae1e2
      abb87(21)=abb87(20)*spbk1e1
      abb87(16)=abb87(21)*abb87(16)
      abb87(22)=spae2l3*spbl5l3*abb87(4)
      abb87(23)=spae1k2*spbe2e1
      abb87(24)=-abb87(23)*abb87(22)
      abb87(25)=abb87(13)*spbl3e1
      abb87(26)=-abb87(20)*abb87(25)
      abb87(24)=abb87(26)+abb87(24)
      abb87(26)=-abb87(9)*abb87(11)
      abb87(27)=abb87(2)**3
      abb87(28)=-abb87(26)*abb87(27)*mT
      abb87(24)=abb87(28)*abb87(24)
      abb87(29)=abb87(27)*abb87(9)
      abb87(30)=abb87(17)*mT
      abb87(29)=abb87(30)+abb87(29)
      abb87(30)=-mT*abb87(11)
      abb87(29)=-abb87(29)*abb87(30)
      abb87(31)=spbl5k1*abb87(29)
      abb87(8)=-abb87(26)*abb87(8)**2
      abb87(32)=abb87(8)*abb87(13)
      abb87(33)=spbl3k1*abb87(32)
      abb87(31)=abb87(33)+abb87(31)
      abb87(33)=spae2k2*abb87(4)
      abb87(34)=abb87(33)*spbe2e1
      abb87(31)=abb87(34)*abb87(31)
      abb87(35)=abb87(12)*spbl5l3
      abb87(36)=abb87(35)*abb87(34)
      abb87(37)=abb87(36)*spak2l3
      abb87(38)=-spbk2k1*abb87(37)
      abb87(31)=abb87(38)+abb87(31)
      abb87(31)=spae1k1*abb87(31)
      abb87(38)=spbl4k2*spae2k2
      abb87(39)=abb87(23)*abb87(38)
      abb87(40)=abb87(29)*abb87(5)
      abb87(41)=abb87(40)*abb87(39)
      abb87(10)=abb87(17)+abb87(10)
      abb87(10)=-abb87(10)*abb87(11)
      abb87(17)=abb87(10)*spbl5k2
      abb87(17)=abb87(17)+abb87(40)
      abb87(42)=abb87(21)*abb87(17)
      abb87(43)=abb87(10)*spbl5e2
      abb87(44)=abb87(43)*spae1e2
      abb87(45)=abb87(44)*spbl4k2
      abb87(46)=-spbk1e1*abb87(45)
      abb87(42)=abb87(46)+abb87(42)
      abb87(42)=spak1k2*abb87(42)
      abb87(46)=spbl5k2*mH**2*abb87(7)*abb87(6)
      abb87(18)=abb87(46)*abb87(18)
      abb87(47)=-spak1k2*abb87(21)
      abb87(39)=-abb87(39)+abb87(47)
      abb87(39)=abb87(39)*abb87(18)
      abb87(47)=abb87(9)*mT
      abb87(27)=abb87(27)*abb87(47)
      abb87(48)=abb87(9)*abb87(2)**4
      abb87(27)=abb87(27)+abb87(48)
      abb87(11)=abb87(11)*abb87(27)
      abb87(27)=spbl5e2*abb87(11)
      abb87(48)=abb87(28)*abb87(13)*spbl3e2
      abb87(27)=abb87(27)+abb87(48)
      abb87(27)=spbl4e1*spae1e2*abb87(27)
      abb87(28)=spbl5l3*abb87(28)*abb87(34)
      abb87(48)=-spbe2e1*abb87(38)*abb87(19)
      abb87(28)=abb87(28)+abb87(48)
      abb87(28)=spae1l3*abb87(28)
      abb87(48)=abb87(14)*spbl3e2
      abb87(49)=abb87(48)*spae1e2
      abb87(50)=abb87(49)*spbl4k2
      abb87(51)=spak1k2*spbk1e1
      abb87(52)=-abb87(51)*abb87(50)
      abb87(11)=-spbl5e1*abb87(11)*abb87(20)
      abb87(11)=abb87(28)+abb87(27)+abb87(11)+abb87(52)+abb87(39)+abb87(41)+abb&
      &87(42)+abb87(31)+abb87(24)+abb87(16)
      abb87(16)=spbl5e1*abb87(10)
      abb87(24)=spbl3e1*abb87(14)
      abb87(16)=abb87(24)+abb87(16)
      abb87(16)=abb87(20)*abb87(16)
      abb87(24)=abb87(44)+abb87(49)
      abb87(27)=-spbl4e1*abb87(24)
      abb87(12)=abb87(22)*abb87(12)
      abb87(22)=abb87(23)*abb87(12)
      abb87(28)=-spae1l3*abb87(36)
      abb87(16)=abb87(28)+abb87(22)+abb87(27)+abb87(16)
      abb87(15)=-abb87(18)+abb87(17)+abb87(15)
      abb87(17)=-spbl4e2*abb87(15)
      abb87(22)=abb87(48)+abb87(43)
      abb87(27)=spbl4k2*abb87(22)
      abb87(17)=abb87(17)+abb87(27)
      abb87(17)=spae1k2*abb87(17)
      abb87(27)=abb87(19)*spae1l3
      abb87(28)=spbl4e2*abb87(27)
      abb87(17)=abb87(28)+abb87(17)
      abb87(28)=abb87(29)*spbl5e1
      abb87(8)=abb87(8)*abb87(25)
      abb87(8)=abb87(28)+abb87(8)
      abb87(25)=-abb87(33)*abb87(8)
      abb87(9)=abb87(9)*abb87(2)
      abb87(9)=abb87(9)+abb87(47)
      abb87(9)=-abb87(9)*abb87(30)
      abb87(28)=abb87(9)*abb87(5)
      abb87(30)=abb87(46)*abb87(26)
      abb87(28)=abb87(28)-abb87(30)
      abb87(30)=-abb87(51)*abb87(28)
      abb87(31)=-abb87(38)*abb87(30)
      abb87(39)=abb87(26)*spbl5l3
      abb87(41)=abb87(38)*abb87(39)
      abb87(42)=spak1l3*spbk1e1
      abb87(43)=-abb87(41)*abb87(42)
      abb87(33)=abb87(35)*abb87(33)
      abb87(35)=spbk2e1*spak2l3*abb87(33)
      abb87(25)=abb87(35)+abb87(43)+abb87(31)+abb87(25)
      abb87(29)=abb87(29)*abb87(34)
      abb87(31)=spak1e2*spbk1e1
      abb87(35)=spbl4e2*abb87(10)*abb87(31)
      abb87(29)=abb87(29)+abb87(35)
      abb87(35)=2.0_ki*spbl4e2
      abb87(35)=abb87(10)*abb87(35)
      abb87(18)=abb87(18)-abb87(40)
      abb87(18)=abb87(23)*abb87(18)
      abb87(23)=spbe2e1*abb87(27)
      abb87(18)=abb87(23)+abb87(18)
      abb87(23)=abb87(39)*abb87(42)
      abb87(23)=abb87(23)+abb87(30)
      abb87(27)=-abb87(22)*abb87(31)
      abb87(22)=-2.0_ki*abb87(22)
      abb87(30)=abb87(34)*abb87(32)
      abb87(32)=abb87(14)*spbl4e2
      abb87(31)=abb87(32)*abb87(31)
      abb87(30)=abb87(30)+abb87(31)
      abb87(31)=2.0_ki*abb87(32)
      abb87(19)=-abb87(20)*abb87(19)
      abb87(32)=-spbe2k1*spae1k1*abb87(33)
      abb87(19)=abb87(19)+abb87(32)
      abb87(32)=-2.0_ki*abb87(33)-abb87(41)
      abb87(8)=-abb87(4)*spae1e2*abb87(8)
      abb87(33)=-spbl5k1*abb87(9)
      abb87(13)=abb87(13)*abb87(26)*mT**2
      abb87(26)=spbl3k1*abb87(13)
      abb87(26)=abb87(33)+abb87(26)
      abb87(26)=spae1k1*abb87(4)*abb87(26)
      abb87(9)=-abb87(4)*abb87(9)
      abb87(13)=abb87(4)*abb87(13)
      abb87(15)=abb87(20)*abb87(15)
      abb87(20)=abb87(12)*spae1k1
      abb87(33)=spbe2k1*abb87(20)
      abb87(15)=abb87(33)-abb87(50)-abb87(45)+abb87(15)
      abb87(33)=abb87(38)*abb87(28)
      abb87(12)=2.0_ki*abb87(12)+abb87(33)
      abb87(33)=spae1k1*abb87(36)
      abb87(20)=-spbe2e1*abb87(20)
      abb87(10)=-abb87(10)*abb87(21)
      abb87(24)=spbk1e1*abb87(24)
      abb87(14)=-abb87(14)*abb87(21)
      R2d87=0.0_ki
      rat2 = rat2 + R2d87
      if (debug_nlo_diagrams) then
          write (logfile,*) "<result name='r2' index='87' value='", &
          & R2d87, "'/>"
      end if
   end subroutine
end module p2_gg_httbar_abbrevd87h12_qp
