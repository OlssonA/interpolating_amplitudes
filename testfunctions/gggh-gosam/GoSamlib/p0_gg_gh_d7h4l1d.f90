module     p0_gg_gh_d7h4l1d
   ! file: /mt/home/sjones/repos/POWHEG-BOX/ggh-gosam/GoSam_POWHEG/Virtual/p0_g &
   ! &g_gh/helicity4d7h4l1d.f90
   ! generator: buildfortran_d.py
   use p0_gg_gh_config, only: ki
   use p0_gg_gh_util, only: cond, d => metric_tensor
   implicit none
   private
   complex(ki), parameter :: i_ = (0.0_ki, 1.0_ki)
   integer, private :: iv0
   integer, private :: iv1
   integer, private :: iv2
   integer, private :: iv3
   integer, private :: iv4
   real(ki), dimension(4), private :: qshift
   public :: derivative
contains
!---#[ function brack_1:
   pure function brack_1(Q, mu2) result(brack)
      use p0_gg_gh_model
      use p0_gg_gh_kinematics
      use p0_gg_gh_color
      use p0_gg_gh_abbrevd7h4
      implicit none
      complex(ki), dimension(4), intent(in) :: Q
      complex(ki), intent(in) :: mu2
      complex(ki), dimension(27) :: acd7
      complex(ki) :: brack
      acd7(1)=dotproduct(k2,qshift)
      acd7(2)=abb7(17)
      acd7(3)=dotproduct(k3,qshift)
      acd7(4)=dotproduct(qshift,spvak1k2)
      acd7(5)=abb7(8)
      acd7(6)=dotproduct(qshift,spvak1k3)
      acd7(7)=abb7(5)
      acd7(8)=dotproduct(qshift,spvak3k2)
      acd7(9)=abb7(7)
      acd7(10)=abb7(16)
      acd7(11)=abb7(15)
      acd7(12)=dotproduct(qshift,qshift)
      acd7(13)=abb7(14)
      acd7(14)=dotproduct(qshift,spvak2k3)
      acd7(15)=abb7(10)
      acd7(16)=dotproduct(qshift,spvak2k1)
      acd7(17)=abb7(12)
      acd7(18)=abb7(11)
      acd7(19)=abb7(13)
      acd7(20)=abb7(6)
      acd7(21)=abb7(9)
      acd7(22)=-acd7(1)-acd7(3)
      acd7(22)=acd7(2)*acd7(22)
      acd7(23)=acd7(6)*acd7(7)
      acd7(24)=acd7(4)*acd7(5)
      acd7(25)=acd7(8)*acd7(9)
      acd7(22)=acd7(25)+acd7(24)-acd7(10)+acd7(23)+acd7(22)
      acd7(22)=acd7(1)*acd7(22)
      acd7(23)=acd7(3)+acd7(12)
      acd7(23)=acd7(9)*acd7(23)
      acd7(24)=acd7(6)*acd7(18)
      acd7(25)=-acd7(4)*acd7(14)*acd7(15)
      acd7(23)=acd7(25)+acd7(24)-acd7(20)+acd7(23)
      acd7(23)=acd7(8)*acd7(23)
      acd7(24)=acd7(12)*acd7(13)
      acd7(25)=-acd7(6)*acd7(19)
      acd7(26)=acd7(4)*acd7(16)*acd7(17)
      acd7(27)=-acd7(3)*acd7(11)
      brack=acd7(21)+acd7(22)+acd7(23)+acd7(24)+acd7(25)+acd7(26)+acd7(27)
   end function brack_1
!---#] function brack_1:
!---#[ function brack_2:
   pure function brack_2(Q, mu2) result(brack)
      use p0_gg_gh_model
      use p0_gg_gh_kinematics
      use p0_gg_gh_color
      use p0_gg_gh_abbrevd7h4
      implicit none
      complex(ki), dimension(4), intent(in) :: Q
      complex(ki), intent(in) :: mu2
      complex(ki), dimension(36) :: acd7
      complex(ki) :: brack
      acd7(1)=k2(iv1)
      acd7(2)=dotproduct(k2,qshift)
      acd7(3)=abb7(17)
      acd7(4)=dotproduct(k3,qshift)
      acd7(5)=dotproduct(qshift,spvak1k2)
      acd7(6)=abb7(8)
      acd7(7)=dotproduct(qshift,spvak1k3)
      acd7(8)=abb7(5)
      acd7(9)=dotproduct(qshift,spvak3k2)
      acd7(10)=abb7(7)
      acd7(11)=abb7(16)
      acd7(12)=k3(iv1)
      acd7(13)=abb7(15)
      acd7(14)=qshift(iv1)
      acd7(15)=abb7(14)
      acd7(16)=spvak1k2(iv1)
      acd7(17)=dotproduct(qshift,spvak2k3)
      acd7(18)=abb7(10)
      acd7(19)=dotproduct(qshift,spvak2k1)
      acd7(20)=abb7(12)
      acd7(21)=spvak1k3(iv1)
      acd7(22)=abb7(11)
      acd7(23)=abb7(13)
      acd7(24)=spvak3k2(iv1)
      acd7(25)=dotproduct(qshift,qshift)
      acd7(26)=abb7(6)
      acd7(27)=spvak2k1(iv1)
      acd7(28)=spvak2k3(iv1)
      acd7(29)=-acd7(21)*acd7(22)
      acd7(30)=acd7(16)*acd7(18)*acd7(17)
      acd7(31)=acd7(5)*acd7(18)
      acd7(32)=acd7(28)*acd7(31)
      acd7(33)=2.0_ki*acd7(14)
      acd7(34)=-acd7(33)-acd7(12)
      acd7(34)=acd7(10)*acd7(34)
      acd7(29)=acd7(34)+acd7(32)+acd7(29)+acd7(30)
      acd7(29)=acd7(9)*acd7(29)
      acd7(30)=2.0_ki*acd7(2)+acd7(4)
      acd7(30)=acd7(3)*acd7(30)
      acd7(32)=-acd7(7)*acd7(8)
      acd7(34)=-acd7(5)*acd7(6)
      acd7(35)=-acd7(9)*acd7(10)
      acd7(30)=acd7(35)+acd7(34)+acd7(11)+acd7(32)+acd7(30)
      acd7(30)=acd7(1)*acd7(30)
      acd7(32)=-acd7(7)*acd7(22)
      acd7(31)=acd7(17)*acd7(31)
      acd7(34)=-acd7(2)-acd7(25)-acd7(4)
      acd7(34)=acd7(10)*acd7(34)
      acd7(31)=acd7(34)+acd7(31)+acd7(26)+acd7(32)
      acd7(31)=acd7(24)*acd7(31)
      acd7(32)=-acd7(21)*acd7(8)
      acd7(34)=-acd7(16)*acd7(6)
      acd7(35)=acd7(3)*acd7(12)
      acd7(32)=acd7(35)+acd7(32)+acd7(34)
      acd7(32)=acd7(2)*acd7(32)
      acd7(34)=-acd7(16)*acd7(19)
      acd7(35)=-acd7(5)*acd7(27)
      acd7(34)=acd7(35)+acd7(34)
      acd7(34)=acd7(20)*acd7(34)
      acd7(33)=-acd7(15)*acd7(33)
      acd7(35)=acd7(21)*acd7(23)
      acd7(36)=acd7(12)*acd7(13)
      brack=acd7(29)+acd7(30)+acd7(31)+acd7(32)+acd7(33)+acd7(34)+acd7(35)+acd7&
      &(36)
   end function brack_2
!---#] function brack_2:
!---#[ function brack_3:
   pure function brack_3(Q, mu2) result(brack)
      use p0_gg_gh_model
      use p0_gg_gh_kinematics
      use p0_gg_gh_color
      use p0_gg_gh_abbrevd7h4
      implicit none
      complex(ki), dimension(4), intent(in) :: Q
      complex(ki), intent(in) :: mu2
      complex(ki), dimension(37) :: acd7
      complex(ki) :: brack
      acd7(1)=d(iv1,iv2)
      acd7(2)=dotproduct(qshift,spvak3k2)
      acd7(3)=abb7(7)
      acd7(4)=abb7(14)
      acd7(5)=k2(iv1)
      acd7(6)=k2(iv2)
      acd7(7)=abb7(17)
      acd7(8)=k3(iv2)
      acd7(9)=spvak3k2(iv2)
      acd7(10)=spvak1k2(iv2)
      acd7(11)=abb7(8)
      acd7(12)=spvak1k3(iv2)
      acd7(13)=abb7(5)
      acd7(14)=k3(iv1)
      acd7(15)=spvak3k2(iv1)
      acd7(16)=spvak1k2(iv1)
      acd7(17)=spvak1k3(iv1)
      acd7(18)=qshift(iv1)
      acd7(19)=qshift(iv2)
      acd7(20)=dotproduct(qshift,spvak2k3)
      acd7(21)=abb7(10)
      acd7(22)=abb7(11)
      acd7(23)=spvak2k3(iv2)
      acd7(24)=dotproduct(qshift,spvak1k2)
      acd7(25)=spvak2k3(iv1)
      acd7(26)=spvak2k1(iv2)
      acd7(27)=abb7(12)
      acd7(28)=spvak2k1(iv1)
      acd7(29)=acd7(13)*acd7(17)
      acd7(30)=-acd7(7)*acd7(14)
      acd7(31)=acd7(16)*acd7(11)
      acd7(29)=acd7(31)+acd7(29)+acd7(30)
      acd7(29)=acd7(6)*acd7(29)
      acd7(30)=-2.0_ki*acd7(6)-acd7(8)
      acd7(30)=acd7(7)*acd7(30)
      acd7(31)=acd7(12)*acd7(13)
      acd7(32)=acd7(10)*acd7(11)
      acd7(30)=acd7(32)+acd7(31)+acd7(30)
      acd7(30)=acd7(5)*acd7(30)
      acd7(31)=2.0_ki*acd7(1)
      acd7(32)=acd7(2)*acd7(31)
      acd7(33)=acd7(6)+2.0_ki*acd7(19)+acd7(8)
      acd7(33)=acd7(15)*acd7(33)
      acd7(34)=acd7(5)+2.0_ki*acd7(18)+acd7(14)
      acd7(34)=acd7(9)*acd7(34)
      acd7(32)=acd7(34)+acd7(32)+acd7(33)
      acd7(32)=acd7(3)*acd7(32)
      acd7(33)=acd7(16)*acd7(26)
      acd7(34)=acd7(10)*acd7(28)
      acd7(33)=acd7(34)+acd7(33)
      acd7(33)=acd7(27)*acd7(33)
      acd7(34)=-acd7(16)*acd7(23)
      acd7(35)=-acd7(10)*acd7(25)
      acd7(34)=acd7(34)+acd7(35)
      acd7(34)=acd7(21)*acd7(2)*acd7(34)
      acd7(35)=-acd7(23)*acd7(24)
      acd7(36)=-acd7(10)*acd7(20)
      acd7(35)=acd7(35)+acd7(36)
      acd7(35)=acd7(21)*acd7(35)
      acd7(36)=acd7(12)*acd7(22)
      acd7(35)=acd7(36)+acd7(35)
      acd7(35)=acd7(15)*acd7(35)
      acd7(36)=-acd7(24)*acd7(25)
      acd7(37)=-acd7(16)*acd7(20)
      acd7(36)=acd7(36)+acd7(37)
      acd7(36)=acd7(21)*acd7(36)
      acd7(37)=acd7(17)*acd7(22)
      acd7(36)=acd7(37)+acd7(36)
      acd7(36)=acd7(9)*acd7(36)
      acd7(31)=acd7(4)*acd7(31)
      brack=acd7(29)+acd7(30)+acd7(31)+acd7(32)+acd7(33)+acd7(34)+acd7(35)+acd7&
      &(36)
   end function brack_3
!---#] function brack_3:
!---#[ function brack_4:
   pure function brack_4(Q, mu2) result(brack)
      use p0_gg_gh_model
      use p0_gg_gh_kinematics
      use p0_gg_gh_color
      use p0_gg_gh_abbrevd7h4
      implicit none
      complex(ki), dimension(4), intent(in) :: Q
      complex(ki), intent(in) :: mu2
      complex(ki), dimension(18) :: acd7
      complex(ki) :: brack
      acd7(1)=d(iv1,iv2)
      acd7(2)=spvak3k2(iv3)
      acd7(3)=abb7(7)
      acd7(4)=d(iv1,iv3)
      acd7(5)=spvak3k2(iv2)
      acd7(6)=d(iv2,iv3)
      acd7(7)=spvak3k2(iv1)
      acd7(8)=spvak1k2(iv2)
      acd7(9)=spvak2k3(iv3)
      acd7(10)=abb7(10)
      acd7(11)=spvak1k2(iv3)
      acd7(12)=spvak2k3(iv2)
      acd7(13)=spvak1k2(iv1)
      acd7(14)=spvak2k3(iv1)
      acd7(15)=acd7(11)*acd7(12)
      acd7(16)=acd7(8)*acd7(9)
      acd7(15)=acd7(15)+acd7(16)
      acd7(15)=acd7(7)*acd7(15)
      acd7(16)=acd7(11)*acd7(14)
      acd7(17)=acd7(9)*acd7(13)
      acd7(16)=acd7(16)+acd7(17)
      acd7(16)=acd7(5)*acd7(16)
      acd7(17)=acd7(12)*acd7(13)
      acd7(18)=acd7(8)*acd7(14)
      acd7(17)=acd7(17)+acd7(18)
      acd7(17)=acd7(2)*acd7(17)
      acd7(15)=acd7(17)+acd7(15)+acd7(16)
      acd7(15)=acd7(10)*acd7(15)
      acd7(16)=-acd7(7)*acd7(6)
      acd7(17)=-acd7(5)*acd7(4)
      acd7(18)=-acd7(2)*acd7(1)
      acd7(16)=acd7(18)+acd7(16)+acd7(17)
      acd7(16)=acd7(3)*acd7(16)
      brack=acd7(15)+2.0_ki*acd7(16)
   end function brack_4
!---#] function brack_4:
!---#[ function brack_5:
   pure function brack_5(Q, mu2) result(brack)
      use p0_gg_gh_model
      use p0_gg_gh_kinematics
      use p0_gg_gh_color
      use p0_gg_gh_abbrevd7h4
      implicit none
      complex(ki), dimension(4), intent(in) :: Q
      complex(ki), intent(in) :: mu2
      complex(ki), dimension(1) :: acd7
      complex(ki) :: brack
      brack=0.0_ki
   end function brack_5
!---#] function brack_5:
!---#[ function derivative:
   function derivative(mu2,i1,i2,i3,i4) result(numerator)
      use p0_gg_gh_globalsl1, only: epspow
      use p0_gg_gh_kinematics
      use p0_gg_gh_abbrevd7h4
      implicit none
      complex(ki), intent(in) :: mu2
      integer, intent(in), optional :: i1
      integer, intent(in), optional :: i2
      integer, intent(in), optional :: i3
      integer, intent(in), optional :: i4
      complex(ki) :: numerator
      complex(ki) :: loc
      integer :: t1
      integer :: deg
      complex(ki), dimension(4), parameter :: Q = (/ (0.0_ki,0.0_ki),(0.0_ki,0.&
      &0_ki),(0.0_ki,0.0_ki),(0.0_ki,0.0_ki)/)
      qshift = k2
      numerator = 0.0_ki
      deg = 0
      if(present(i1)) then
          iv1=i1
          deg=1
      else
          iv1=1
      end if
      if(present(i2)) then
          iv2=i2
          deg=2
      else
          iv2=1
      end if
      if(present(i3)) then
          iv3=i3
          deg=3
      else
          iv3=1
      end if
      if(present(i4)) then
          iv4=i4
          deg=4
      else
          iv4=1
      end if
      t1 = 0
      if(deg.eq.0) then
         numerator = cond(epspow.eq.t1,brack_1,Q,mu2)
         return
      end if
      if(deg.eq.1) then
         numerator = cond(epspow.eq.t1,brack_2,Q,mu2)
         return
      end if
      if(deg.eq.2) then
         numerator = cond(epspow.eq.t1,brack_3,Q,mu2)
         return
      end if
      if(deg.eq.3) then
         numerator = cond(epspow.eq.t1,brack_4,Q,mu2)
         return
      end if
      if(deg.eq.4) then
         numerator = cond(epspow.eq.t1,brack_5,Q,mu2)
         return
      end if
   end function derivative
!---#] function derivative:
end module     p0_gg_gh_d7h4l1d
