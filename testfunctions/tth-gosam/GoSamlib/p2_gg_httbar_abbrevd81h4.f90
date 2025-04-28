module     p2_gg_httbar_abbrevd81h4
   use p2_gg_httbar_config, only: ki
   use p2_gg_httbar_kinematics, only: epstensor
   use p2_gg_httbar_globalsh4
   implicit none
   private
   complex(ki), dimension(58), public :: abb81
   complex(ki), public :: R2d81
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
      abb81(1)=1.0_ki/(-mT**2+es34)
      abb81(2)=sqrt(mT**2)
      abb81(3)=NC**(-1)
      abb81(4)=spak2l3**(-1)
      abb81(5)=spbl3k2**(-1)
      abb81(6)=spak2l4**(-1)
      abb81(7)=spbl5k2**(-1)
      abb81(8)=c2*abb81(3)
      abb81(8)=abb81(8)-c3
      abb81(9)=i_*TR*e*gHT*abb81(1)*gs**4
      abb81(10)=abb81(9)*abb81(2)
      abb81(11)=-abb81(10)*abb81(8)
      abb81(12)=mT**2
      abb81(13)=-abb81(12)*abb81(11)
      abb81(14)=spak2l3*abb81(6)
      abb81(15)=abb81(14)*spbl3e1
      abb81(16)=abb81(13)*abb81(15)
      abb81(17)=abb81(2)**2
      abb81(18)=abb81(17)*abb81(9)
      abb81(19)=abb81(10)*mT
      abb81(18)=abb81(19)+abb81(18)
      abb81(19)=-mT*abb81(8)
      abb81(18)=-abb81(18)*abb81(19)
      abb81(20)=abb81(18)*spbl4e1
      abb81(16)=abb81(16)+abb81(20)
      abb81(20)=spae1e2*abb81(7)
      abb81(21)=abb81(20)*spbk2e2
      abb81(22)=abb81(21)*abb81(16)
      abb81(23)=abb81(18)*abb81(6)
      abb81(24)=abb81(23)*spae1l5
      abb81(25)=spae2k2*spbe2e1
      abb81(26)=abb81(24)*abb81(25)
      abb81(27)=abb81(11)*spbe2e1
      abb81(28)=abb81(27)*spae2k2
      abb81(29)=abb81(28)*spae1l5
      abb81(30)=spbl4k2*mH**2*abb81(5)*abb81(4)
      abb81(31)=abb81(29)*abb81(30)
      abb81(22)=-abb81(22)+abb81(26)-abb81(31)
      abb81(26)=-es12*abb81(22)
      abb81(18)=abb81(18)*spbk2e1
      abb81(31)=abb81(18)*spbl4k1
      abb81(32)=-spak1k2*abb81(31)
      abb81(33)=abb81(2)**3
      abb81(34)=abb81(33)*abb81(9)
      abb81(35)=abb81(8)*abb81(34)
      abb81(36)=abb81(35)*abb81(12)
      abb81(37)=-abb81(36)*abb81(15)
      abb81(32)=abb81(37)+abb81(32)
      abb81(32)=abb81(21)*abb81(32)
      abb81(37)=spae1l5*abb81(6)*abb81(25)
      abb81(38)=-spbl4e1*abb81(21)
      abb81(39)=spbl4e2*spbk2e1*abb81(20)
      abb81(37)=abb81(39)+abb81(37)+abb81(38)
      abb81(38)=abb81(9)*mT
      abb81(33)=abb81(33)*abb81(38)
      abb81(9)=abb81(9)*abb81(2)**4
      abb81(9)=abb81(9)+abb81(33)
      abb81(9)=-abb81(9)*abb81(19)
      abb81(19)=abb81(9)*abb81(37)
      abb81(33)=abb81(30)*spae1l5
      abb81(37)=abb81(35)*abb81(25)*abb81(33)
      abb81(39)=spae1k2*abb81(6)
      abb81(9)=-abb81(9)*abb81(39)
      abb81(40)=abb81(30)*spae1k2
      abb81(41)=-abb81(35)*abb81(40)
      abb81(9)=abb81(9)+abb81(41)
      abb81(41)=spae2l5*spbe2e1
      abb81(9)=abb81(9)*abb81(41)
      abb81(42)=abb81(23)*spae1k2
      abb81(25)=abb81(42)*abb81(25)
      abb81(43)=-abb81(28)*abb81(40)
      abb81(43)=abb81(25)+abb81(43)
      abb81(44)=spbk2k1*spak1l5
      abb81(43)=abb81(43)*abb81(44)
      abb81(35)=abb81(35)*spbl4l3
      abb81(45)=-abb81(41)*abb81(35)
      abb81(44)=-spbl4l3*abb81(28)*abb81(44)
      abb81(44)=abb81(45)+abb81(44)
      abb81(44)=spae1l3*abb81(44)
      abb81(17)=abb81(38)*abb81(17)
      abb81(34)=abb81(34)+abb81(17)
      abb81(34)=-abb81(34)*abb81(8)
      abb81(45)=abb81(34)*abb81(41)
      abb81(46)=spae1k1*spbl4k1
      abb81(47)=abb81(46)*abb81(45)
      abb81(48)=abb81(34)*spbl4e1
      abb81(17)=-abb81(17)*abb81(8)
      abb81(15)=abb81(17)*abb81(15)
      abb81(15)=abb81(15)+abb81(48)
      abb81(48)=spae1e2*abb81(15)
      abb81(49)=spak2l5*spbk2e2
      abb81(50)=abb81(48)*abb81(49)
      abb81(34)=-abb81(12)*abb81(34)
      abb81(51)=abb81(6)*abb81(34)*abb81(21)
      abb81(52)=abb81(17)*abb81(21)
      abb81(53)=abb81(52)*abb81(30)
      abb81(51)=abb81(53)-abb81(51)
      abb81(53)=-spbk1e1*spak1k2*abb81(51)
      abb81(41)=abb81(14)*abb81(41)*abb81(17)
      abb81(54)=spae1k1*abb81(41)
      abb81(55)=abb81(14)*spbk2e1
      abb81(13)=abb81(55)*abb81(13)
      abb81(56)=abb81(13)*abb81(21)
      abb81(57)=-spak1k2*abb81(56)
      abb81(54)=abb81(57)+abb81(54)
      abb81(54)=spbl3k1*abb81(54)
      abb81(57)=spbk2k1*abb81(29)
      abb81(58)=-spbk1e1*abb81(52)
      abb81(57)=abb81(57)+abb81(58)
      abb81(57)=spak1l3*spbl4l3*abb81(57)
      abb81(36)=spbl3e2*abb81(36)*abb81(20)*abb81(55)
      abb81(55)=spae1l5*spbe2e1
      abb81(35)=spae2l3*abb81(55)*abb81(35)
      abb81(9)=abb81(35)+abb81(36)+abb81(57)+abb81(26)+abb81(54)+abb81(53)+abb8&
      &1(50)+abb81(47)+abb81(44)+abb81(43)+abb81(9)+abb81(37)+abb81(19)+abb81(3&
      &2)
      abb81(19)=2.0_ki*abb81(22)
      abb81(26)=spbl4e2*abb81(18)
      abb81(32)=spbl3e2*abb81(13)
      abb81(26)=abb81(32)+abb81(26)
      abb81(26)=abb81(20)*abb81(26)
      abb81(32)=spae1l3*spbl4l3
      abb81(35)=spae2l5*abb81(32)
      abb81(36)=spbl4l3*spae1l5
      abb81(37)=-spae2l3*abb81(36)
      abb81(35)=abb81(37)+abb81(35)
      abb81(35)=abb81(27)*abb81(35)
      abb81(37)=abb81(40)*abb81(27)
      abb81(43)=abb81(42)*spbe2e1
      abb81(37)=abb81(37)-abb81(43)
      abb81(43)=spae2l5*abb81(37)
      abb81(22)=abb81(43)+abb81(22)+abb81(35)+abb81(26)
      abb81(26)=abb81(40)*abb81(11)
      abb81(35)=abb81(11)*spbl4l3
      abb81(43)=abb81(35)*spae1l3
      abb81(26)=abb81(43)+abb81(26)-abb81(42)
      abb81(42)=spak1l5*abb81(26)
      abb81(11)=abb81(11)*abb81(33)
      abb81(11)=abb81(11)-abb81(24)
      abb81(24)=-spak1k2*abb81(11)
      abb81(35)=abb81(35)*spae1l5
      abb81(44)=-spak1l3*abb81(35)
      abb81(24)=abb81(44)+abb81(24)+abb81(42)
      abb81(24)=spbe2k1*abb81(24)
      abb81(40)=abb81(32)+abb81(40)
      abb81(17)=abb81(40)*abb81(17)
      abb81(34)=abb81(34)*abb81(39)
      abb81(17)=abb81(17)-abb81(34)
      abb81(34)=spbk2e2*abb81(7)
      abb81(39)=abb81(34)*abb81(17)
      abb81(10)=abb81(38)+abb81(10)
      abb81(10)=-abb81(10)*abb81(8)
      abb81(42)=abb81(46)*abb81(10)
      abb81(44)=abb81(49)*abb81(42)
      abb81(8)=-abb81(38)*abb81(8)
      abb81(14)=abb81(8)*abb81(14)
      abb81(38)=abb81(49)*abb81(14)
      abb81(46)=spbl3k1*spae1k1
      abb81(47)=abb81(38)*abb81(46)
      abb81(24)=abb81(47)+abb81(44)+abb81(39)+abb81(24)
      abb81(39)=abb81(13)*spbl3k1
      abb81(31)=abb81(39)+abb81(31)
      abb81(39)=abb81(7)*abb81(31)
      abb81(44)=abb81(16)*abb81(7)
      abb81(47)=-spbk2k1*abb81(44)
      abb81(39)=abb81(47)+abb81(39)
      abb81(39)=spak1e2*abb81(39)
      abb81(15)=-spae2l5*abb81(15)
      abb81(15)=abb81(39)+abb81(15)
      abb81(39)=spbk2e1*spae2k2
      abb81(43)=-abb81(39)*abb81(43)
      abb81(43)=abb81(43)-abb81(48)
      abb81(46)=-abb81(14)*abb81(46)
      abb81(26)=abb81(46)-2.0_ki*abb81(26)-abb81(42)
      abb81(42)=2.0_ki*abb81(7)
      abb81(46)=-abb81(18)*abb81(42)
      abb81(47)=abb81(10)*abb81(49)
      abb81(13)=-abb81(13)*abb81(42)
      abb81(42)=abb81(39)*abb81(35)
      abb81(35)=2.0_ki*abb81(35)
      abb81(48)=-spbl4l3*abb81(52)
      abb81(17)=spbe2e1*abb81(7)*abb81(17)
      abb81(34)=abb81(34)*spae1k2*abb81(16)
      abb81(17)=abb81(17)+abb81(34)
      abb81(12)=abb81(7)*abb81(6)*abb81(10)*abb81(12)
      abb81(8)=abb81(8)*abb81(7)
      abb81(30)=abb81(8)*abb81(30)
      abb81(12)=abb81(12)+abb81(30)
      abb81(30)=spak1k2*abb81(12)
      abb81(8)=abb81(8)*spbl4l3
      abb81(34)=spak1l3*abb81(8)
      abb81(30)=abb81(34)+abb81(30)
      abb81(30)=spbk1e1*abb81(30)
      abb81(30)=2.0_ki*abb81(44)+abb81(30)
      abb81(34)=abb81(39)*abb81(11)
      abb81(11)=2.0_ki*abb81(11)
      abb81(32)=-spak1l5*abb81(32)
      abb81(36)=spak1l3*abb81(36)
      abb81(32)=abb81(36)+abb81(32)
      abb81(32)=abb81(27)*abb81(32)
      abb81(23)=-abb81(55)*abb81(23)
      abb81(27)=abb81(27)*abb81(33)
      abb81(23)=abb81(23)+abb81(27)
      abb81(23)=spak1k2*abb81(23)
      abb81(27)=-spak1l5*abb81(37)
      abb81(23)=abb81(23)+abb81(27)+abb81(32)
      abb81(16)=spbk2k1*abb81(16)
      abb81(16)=abb81(16)-abb81(31)
      abb81(16)=abb81(20)*abb81(16)
      abb81(20)=abb81(28)*abb81(40)
      abb81(20)=-abb81(25)+abb81(20)
      abb81(25)=-spbl4l3*abb81(29)
      abb81(18)=abb81(21)*abb81(18)
      R2d81=0.0_ki
      rat2 = rat2 + R2d81
      if (debug_nlo_diagrams) then
          write (logfile,*) "<result name='r2' index='81' value='", &
          & R2d81, "'/>"
      end if
   end subroutine
end module p2_gg_httbar_abbrevd81h4
