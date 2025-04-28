module     p2_gg_httbar_abbrevd87h0
   use p2_gg_httbar_config, only: ki
   use p2_gg_httbar_kinematics, only: epstensor
   use p2_gg_httbar_globalsh0
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
      use p2_gg_httbar_kinematics
      use p2_gg_httbar_model
      use p2_gg_httbar_color, only: TR
      use p2_gg_httbar_globalsl1, only: epspow
      implicit none
      abb87(1)=1.0_ki/(mH**2+mT**2-es34-es45+es12)
      abb87(2)=sqrt(mT**2)
      abb87(3)=NC**(-1)
      abb87(4)=spbl5k2**(-1)
      abb87(5)=spbl4k2**(-1)
      abb87(6)=spak2l3**(-1)
      abb87(7)=spbl3k2**(-1)
      abb87(8)=mT*abb87(2)
      abb87(9)=i_*TR*e*gHT*abb87(1)*gs**4
      abb87(10)=abb87(8)*abb87(9)
      abb87(11)=c1*abb87(3)
      abb87(11)=abb87(11)-c3
      abb87(12)=-abb87(10)*abb87(11)
      abb87(13)=spbl3k2*abb87(4)
      abb87(14)=abb87(12)*abb87(13)
      abb87(15)=abb87(14)*spak2l3
      abb87(16)=-spbk2k1*abb87(15)
      abb87(17)=abb87(9)*abb87(2)**2
      abb87(18)=-abb87(17)*abb87(11)
      abb87(19)=abb87(18)*spal3l5
      abb87(20)=spbl3k1*abb87(19)
      abb87(16)=abb87(20)+abb87(16)
      abb87(20)=spae2l4*spbe2e1
      abb87(21)=abb87(20)*spae1k1
      abb87(16)=abb87(21)*abb87(16)
      abb87(22)=spbl3e2*spal3l5*abb87(5)
      abb87(23)=spbk2e1*spae1e2
      abb87(24)=abb87(23)*abb87(22)
      abb87(25)=abb87(13)*spae1l3
      abb87(26)=abb87(20)*abb87(25)
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
      abb87(31)=-spak1l5*abb87(29)
      abb87(8)=-abb87(26)*abb87(8)**2
      abb87(32)=abb87(8)*abb87(13)
      abb87(33)=-spak1l3*abb87(32)
      abb87(31)=abb87(33)+abb87(31)
      abb87(33)=spbk2e2*abb87(5)
      abb87(34)=abb87(33)*spae1e2
      abb87(31)=abb87(34)*abb87(31)
      abb87(35)=abb87(12)*spal3l5
      abb87(36)=abb87(35)*abb87(34)
      abb87(37)=abb87(36)*spbl3k2
      abb87(38)=spak1k2*abb87(37)
      abb87(31)=abb87(38)+abb87(31)
      abb87(31)=spbk1e1*abb87(31)
      abb87(38)=spak2l4*spbk2e2
      abb87(39)=abb87(23)*abb87(38)
      abb87(40)=abb87(29)*abb87(4)
      abb87(41)=-abb87(40)*abb87(39)
      abb87(10)=abb87(17)+abb87(10)
      abb87(10)=-abb87(10)*abb87(11)
      abb87(17)=abb87(10)*spak2l5
      abb87(17)=abb87(17)+abb87(40)
      abb87(42)=-abb87(21)*abb87(17)
      abb87(43)=abb87(10)*spae2l5
      abb87(44)=abb87(43)*spbe2e1
      abb87(45)=abb87(44)*spak2l4
      abb87(46)=spae1k1*abb87(45)
      abb87(42)=abb87(46)+abb87(42)
      abb87(42)=spbk2k1*abb87(42)
      abb87(46)=spak2l5*mH**2*abb87(7)*abb87(6)
      abb87(18)=abb87(46)*abb87(18)
      abb87(47)=spbk2k1*abb87(21)
      abb87(39)=abb87(39)+abb87(47)
      abb87(39)=abb87(39)*abb87(18)
      abb87(47)=abb87(9)*mT
      abb87(27)=abb87(27)*abb87(47)
      abb87(48)=abb87(9)*abb87(2)**4
      abb87(27)=abb87(27)+abb87(48)
      abb87(11)=abb87(11)*abb87(27)
      abb87(27)=-spae2l5*abb87(11)
      abb87(48)=-abb87(28)*abb87(13)*spae2l3
      abb87(27)=abb87(27)+abb87(48)
      abb87(27)=spae1l4*spbe2e1*abb87(27)
      abb87(28)=-spal3l5*abb87(28)*abb87(34)
      abb87(48)=spae1e2*abb87(38)*abb87(19)
      abb87(28)=abb87(28)+abb87(48)
      abb87(28)=spbl3e1*abb87(28)
      abb87(48)=abb87(14)*spae2l3
      abb87(49)=abb87(48)*spbe2e1
      abb87(50)=abb87(49)*spak2l4
      abb87(51)=spbk2k1*spae1k1
      abb87(52)=abb87(51)*abb87(50)
      abb87(11)=spae1l5*abb87(11)*abb87(20)
      abb87(11)=abb87(28)+abb87(27)+abb87(11)+abb87(52)+abb87(39)+abb87(41)+abb&
      &87(42)+abb87(31)+abb87(24)+abb87(16)
      abb87(16)=-spae1l5*abb87(10)
      abb87(24)=-spae1l3*abb87(14)
      abb87(16)=abb87(24)+abb87(16)
      abb87(16)=abb87(20)*abb87(16)
      abb87(24)=abb87(44)+abb87(49)
      abb87(27)=spae1l4*abb87(24)
      abb87(12)=abb87(22)*abb87(12)
      abb87(22)=-abb87(23)*abb87(12)
      abb87(28)=spbl3e1*abb87(36)
      abb87(16)=abb87(28)+abb87(22)+abb87(27)+abb87(16)
      abb87(22)=abb87(29)*spae1l5
      abb87(8)=abb87(8)*abb87(25)
      abb87(8)=abb87(22)+abb87(8)
      abb87(22)=abb87(33)*abb87(8)
      abb87(9)=abb87(9)*abb87(2)
      abb87(9)=abb87(9)+abb87(47)
      abb87(9)=-abb87(9)*abb87(30)
      abb87(25)=abb87(9)*abb87(4)
      abb87(27)=abb87(46)*abb87(26)
      abb87(25)=abb87(25)-abb87(27)
      abb87(27)=-abb87(51)*abb87(25)
      abb87(28)=abb87(38)*abb87(27)
      abb87(30)=abb87(26)*spal3l5
      abb87(31)=abb87(38)*abb87(30)
      abb87(39)=spbl3k1*spae1k1
      abb87(41)=abb87(31)*abb87(39)
      abb87(33)=abb87(35)*abb87(33)
      abb87(35)=-spae1k2*spbl3k2*abb87(33)
      abb87(22)=abb87(35)+abb87(41)+abb87(28)+abb87(22)
      abb87(15)=-abb87(18)+abb87(17)+abb87(15)
      abb87(17)=spae2l4*abb87(15)
      abb87(28)=abb87(48)+abb87(43)
      abb87(35)=-spak2l4*abb87(28)
      abb87(17)=abb87(17)+abb87(35)
      abb87(17)=spbk2e1*abb87(17)
      abb87(35)=abb87(19)*spbl3e1
      abb87(41)=-spae2l4*abb87(35)
      abb87(17)=abb87(41)+abb87(17)
      abb87(29)=-abb87(29)*abb87(34)
      abb87(41)=spbe2k1*spae1k1
      abb87(42)=-spae2l4*abb87(10)*abb87(41)
      abb87(29)=abb87(29)+abb87(42)
      abb87(42)=2.0_ki*spae2l4
      abb87(42)=-abb87(10)*abb87(42)
      abb87(18)=abb87(18)-abb87(40)
      abb87(18)=-abb87(23)*abb87(18)
      abb87(23)=-spae1e2*abb87(35)
      abb87(18)=abb87(23)+abb87(18)
      abb87(23)=-abb87(30)*abb87(39)
      abb87(23)=abb87(23)-abb87(27)
      abb87(27)=abb87(28)*abb87(41)
      abb87(28)=2.0_ki*abb87(28)
      abb87(19)=abb87(20)*abb87(19)
      abb87(35)=spak1e2*spbk1e1*abb87(33)
      abb87(19)=abb87(19)+abb87(35)
      abb87(31)=2.0_ki*abb87(33)+abb87(31)
      abb87(32)=-abb87(34)*abb87(32)
      abb87(33)=abb87(14)*spae2l4
      abb87(34)=-abb87(33)*abb87(41)
      abb87(32)=abb87(32)+abb87(34)
      abb87(33)=-2.0_ki*abb87(33)
      abb87(8)=abb87(5)*spbe2e1*abb87(8)
      abb87(34)=spak1l5*abb87(9)
      abb87(13)=abb87(13)*abb87(26)*mT**2
      abb87(26)=-spak1l3*abb87(13)
      abb87(26)=abb87(34)+abb87(26)
      abb87(26)=spbk1e1*abb87(5)*abb87(26)
      abb87(9)=abb87(5)*abb87(9)
      abb87(13)=-abb87(5)*abb87(13)
      abb87(15)=-abb87(20)*abb87(15)
      abb87(20)=abb87(12)*spbk1e1
      abb87(34)=-spak1e2*abb87(20)
      abb87(15)=abb87(34)+abb87(50)+abb87(45)+abb87(15)
      abb87(34)=-abb87(38)*abb87(25)
      abb87(12)=-2.0_ki*abb87(12)+abb87(34)
      abb87(10)=abb87(10)*abb87(21)
      abb87(24)=-spae1k1*abb87(24)
      abb87(14)=abb87(14)*abb87(21)
      abb87(21)=-spbk1e1*abb87(36)
      abb87(20)=spae1e2*abb87(20)
      R2d87=0.0_ki
      rat2 = rat2 + R2d87
      if (debug_nlo_diagrams) then
          write (logfile,*) "<result name='r2' index='87' value='", &
          & R2d87, "'/>"
      end if
   end subroutine
end module p2_gg_httbar_abbrevd87h0
