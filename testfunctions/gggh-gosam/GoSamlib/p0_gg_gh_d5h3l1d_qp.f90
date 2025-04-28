module     p0_gg_gh_d5h3l1d_qp
   ! file: /mt/home/sjones/repos/POWHEG-BOX/ggh-gosam/GoSam_POWHEG/Virtual/p0_g &
   ! &g_gh/helicity3d5h3l1d_qp.f90
   ! generator: buildfortran_d.py
   use p0_gg_gh_config, only: ki => ki_qp
   use p0_gg_gh_util_qp, only: cond, d => metric_tensor
   implicit none
   private
   complex(ki), parameter :: i_ = (0.0_ki, 1.0_ki)
   integer, private :: iv0
   integer, private :: iv1
   integer, private :: iv2
   integer, private :: iv3
   real(ki), dimension(4), private :: qshift
   public :: derivative
contains
!---#[ function brack_1:
   pure function brack_1(Q, mu2) result(brack)
      use p0_gg_gh_model_qp
      use p0_gg_gh_kinematics_qp
      use p0_gg_gh_color_qp
      use p0_gg_gh_abbrevd5h3_qp
      implicit none
      complex(ki), dimension(4), intent(in) :: Q
      complex(ki), intent(in) :: mu2
      complex(ki), dimension(24) :: acd5
      complex(ki) :: brack
      acd5(1)=dotproduct(k2,qshift)
      acd5(2)=dotproduct(qshift,spvak2k1)
      acd5(3)=abb5(9)
      acd5(4)=abb5(15)
      acd5(5)=dotproduct(qshift,qshift)
      acd5(6)=abb5(16)
      acd5(7)=abb5(11)
      acd5(8)=dotproduct(qshift,spvak2k3)
      acd5(9)=dotproduct(qshift,spvak3k1)
      acd5(10)=abb5(18)
      acd5(11)=abb5(14)
      acd5(12)=abb5(7)
      acd5(13)=dotproduct(qshift,spvak2l4)
      acd5(14)=abb5(8)
      acd5(15)=dotproduct(qshift,spval4k1)
      acd5(16)=abb5(10)
      acd5(17)=abb5(13)
      acd5(18)=acd5(3)*acd5(1)
      acd5(18)=-acd5(7)+acd5(18)
      acd5(18)=acd5(2)*acd5(18)
      acd5(19)=acd5(10)*acd5(8)
      acd5(19)=-acd5(12)+acd5(19)
      acd5(19)=acd5(9)*acd5(19)
      acd5(20)=-acd5(4)*acd5(1)
      acd5(21)=acd5(6)*acd5(5)
      acd5(22)=-acd5(11)*acd5(8)
      acd5(23)=-acd5(14)*acd5(13)
      acd5(24)=-acd5(16)*acd5(15)
      brack=acd5(17)+acd5(18)+acd5(19)+acd5(20)+acd5(21)+acd5(22)+acd5(23)+acd5&
      &(24)
   end function brack_1
!---#] function brack_1:
!---#[ function brack_2:
   pure function brack_2(Q, mu2) result(brack)
      use p0_gg_gh_model_qp
      use p0_gg_gh_kinematics_qp
      use p0_gg_gh_color_qp
      use p0_gg_gh_abbrevd5h3_qp
      implicit none
      complex(ki), dimension(4), intent(in) :: Q
      complex(ki), intent(in) :: mu2
      complex(ki), dimension(27) :: acd5
      complex(ki) :: brack
      acd5(1)=k2(iv1)
      acd5(2)=dotproduct(qshift,spvak2k1)
      acd5(3)=abb5(9)
      acd5(4)=abb5(15)
      acd5(5)=qshift(iv1)
      acd5(6)=abb5(16)
      acd5(7)=spvak2k1(iv1)
      acd5(8)=dotproduct(k2,qshift)
      acd5(9)=abb5(11)
      acd5(10)=spvak2k3(iv1)
      acd5(11)=dotproduct(qshift,spvak3k1)
      acd5(12)=abb5(18)
      acd5(13)=abb5(14)
      acd5(14)=spvak3k1(iv1)
      acd5(15)=dotproduct(qshift,spvak2k3)
      acd5(16)=abb5(7)
      acd5(17)=spvak2l4(iv1)
      acd5(18)=abb5(8)
      acd5(19)=spval4k1(iv1)
      acd5(20)=abb5(10)
      acd5(21)=acd5(2)*acd5(3)
      acd5(21)=acd5(21)-acd5(4)
      acd5(21)=acd5(1)*acd5(21)
      acd5(22)=acd5(8)*acd5(3)
      acd5(22)=-acd5(9)+acd5(22)
      acd5(22)=acd5(7)*acd5(22)
      acd5(23)=acd5(11)*acd5(12)
      acd5(23)=-acd5(13)+acd5(23)
      acd5(23)=acd5(10)*acd5(23)
      acd5(24)=acd5(15)*acd5(12)
      acd5(24)=-acd5(16)+acd5(24)
      acd5(24)=acd5(14)*acd5(24)
      acd5(25)=acd5(6)*acd5(5)
      acd5(26)=-acd5(18)*acd5(17)
      acd5(27)=-acd5(20)*acd5(19)
      brack=acd5(21)+acd5(22)+acd5(23)+acd5(24)+2.0_ki*acd5(25)+acd5(26)+acd5(2&
      &7)
   end function brack_2
!---#] function brack_2:
!---#[ function brack_3:
   pure function brack_3(Q, mu2) result(brack)
      use p0_gg_gh_model_qp
      use p0_gg_gh_kinematics_qp
      use p0_gg_gh_color_qp
      use p0_gg_gh_abbrevd5h3_qp
      implicit none
      complex(ki), dimension(4), intent(in) :: Q
      complex(ki), intent(in) :: mu2
      complex(ki), dimension(15) :: acd5
      complex(ki) :: brack
      acd5(1)=d(iv1,iv2)
      acd5(2)=abb5(16)
      acd5(3)=k2(iv1)
      acd5(4)=spvak2k1(iv2)
      acd5(5)=abb5(9)
      acd5(6)=k2(iv2)
      acd5(7)=spvak2k1(iv1)
      acd5(8)=spvak2k3(iv1)
      acd5(9)=spvak3k1(iv2)
      acd5(10)=abb5(18)
      acd5(11)=spvak2k3(iv2)
      acd5(12)=spvak3k1(iv1)
      acd5(13)=acd5(4)*acd5(3)
      acd5(14)=acd5(7)*acd5(6)
      acd5(13)=acd5(14)+acd5(13)
      acd5(13)=acd5(5)*acd5(13)
      acd5(14)=acd5(9)*acd5(8)
      acd5(15)=acd5(12)*acd5(11)
      acd5(14)=acd5(15)+acd5(14)
      acd5(14)=acd5(10)*acd5(14)
      acd5(15)=acd5(2)*acd5(1)
      brack=acd5(13)+acd5(14)+2.0_ki*acd5(15)
   end function brack_3
!---#] function brack_3:
!---#[ function brack_4:
   pure function brack_4(Q, mu2) result(brack)
      use p0_gg_gh_model_qp
      use p0_gg_gh_kinematics_qp
      use p0_gg_gh_color_qp
      use p0_gg_gh_abbrevd5h3_qp
      implicit none
      complex(ki), dimension(4), intent(in) :: Q
      complex(ki), intent(in) :: mu2
      complex(ki), dimension(1) :: acd5
      complex(ki) :: brack
      brack=0.0_ki
   end function brack_4
!---#] function brack_4:
!---#[ function derivative:
   function derivative(mu2,i1,i2,i3) result(numerator)
      use p0_gg_gh_globalsl1_qp, only: epspow
      use p0_gg_gh_kinematics_qp
      use p0_gg_gh_abbrevd5h3_qp
      implicit none
      complex(ki), intent(in) :: mu2
      integer, intent(in), optional :: i1
      integer, intent(in), optional :: i2
      integer, intent(in), optional :: i3
      complex(ki) :: numerator
      complex(ki) :: loc
      integer :: t1
      integer :: deg
      complex(ki), dimension(4), parameter :: Q = (/ (0.0_ki,0.0_ki),(0.0_ki,0.&
      &0_ki),(0.0_ki,0.0_ki),(0.0_ki,0.0_ki)/)
      qshift = -k4-k3
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
   end function derivative
!---#] function derivative:
end module     p0_gg_gh_d5h3l1d_qp
